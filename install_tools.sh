#!/usr/bin/env bash
# install_tools.sh — Install all security tools used by Mythos Engine
# Supports: Fedora/RHEL (dnf), Debian/Ubuntu (apt), and direct Go/binary installs.
# Run as root or with sudo.

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[*]${RESET} $*"; }
success() { echo -e "${GREEN}[+]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
error()   { echo -e "${RED}[-]${RESET} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}"; }

FAILED=()

# ── Detect package manager ─────────────────────────────────────────────────
if command -v dnf &>/dev/null; then
    PKG_MGR="dnf"
    PKG_INSTALL="dnf install -y --quiet"
    PKG_UPDATE="dnf check-update -q || true"
elif command -v apt-get &>/dev/null; then
    PKG_MGR="apt"
    PKG_INSTALL="apt-get install -y -qq"
    PKG_UPDATE="apt-get update -qq"
else
    error "Unsupported distro — no dnf or apt-get found."
    exit 1
fi

info "Detected package manager: ${PKG_MGR}"
GO_BIN_DIR="/usr/local/go/bin"
INSTALL_DIR="/usr/local/bin"

# ── Helper: install a binary tool, track failures ─────────────────────────
install_pkg() {
    local name="$1"; shift
    if $PKG_INSTALL "$@" &>/dev/null 2>&1; then
        success "${name}"
    else
        warn "${name} — package install failed, may need manual install"
        FAILED+=("$name")
    fi
}

install_go_tool() {
    local name="$1"
    local pkg="$2"
    local bin="${3:-$name}"
    if command -v "$bin" &>/dev/null; then
        success "${name} (already installed)"
        return
    fi
    if ! command -v go &>/dev/null; then
        warn "${name} — Go not installed, skipping"
        FAILED+=("$name (needs Go)")
        return
    fi
    info "  go install ${pkg}..."
    if GOPATH=/usr/local go install "${pkg}" &>/dev/null 2>&1; then
        # Link to INSTALL_DIR if needed
        local built="${GO_BIN_DIR}/${bin}"
        if [[ -f "$built" ]] && [[ ! -f "${INSTALL_DIR}/${bin}" ]]; then
            ln -sf "$built" "${INSTALL_DIR}/${bin}"
        fi
        success "${name}"
    else
        warn "${name} — go install failed"
        FAILED+=("$name")
    fi
}

install_binary() {
    local name="$1"
    local url="$2"
    local dest="${3:-${INSTALL_DIR}/${name}}"
    if command -v "$name" &>/dev/null; then
        success "${name} (already installed)"
        return
    fi
    info "  Downloading ${name}..."
    if curl -sSfL "$url" -o "$dest" 2>/dev/null && chmod +x "$dest"; then
        success "${name}"
    else
        warn "${name} — download failed from ${url}"
        FAILED+=("$name")
    fi
}

# ── 1. System update ────────────────────────────────────────────────────────
header "System Update"
info "Running package index update..."
$PKG_UPDATE
success "Package index updated"

# ── 2. Core build dependencies ──────────────────────────────────────────────
header "Build Dependencies"
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "build-essentials" gcc make git curl wget unzip tar python3 python3-pip
    install_pkg "development libs" openssl-devel libffi-devel
else
    install_pkg "build-essentials" build-essential git curl wget unzip tar python3 python3-pip libssl-dev libffi-dev
fi

# ── 3. Go (needed for many tools) ───────────────────────────────────────────
header "Go Language Runtime"
if command -v go &>/dev/null; then
    success "Go $(go version | awk '{print $3}') already installed"
else
    GO_VERSION="1.22.3"
    ARCH=$(uname -m); [[ "$ARCH" == "x86_64" ]] && ARCH="amd64" || ARCH="arm64"
    GO_URL="https://go.dev/dl/go${GO_VERSION}.linux-${ARCH}.tar.gz"
    info "Installing Go ${GO_VERSION}..."
    curl -sSfL "$GO_URL" | tar -C /usr/local -xz 2>/dev/null || { warn "Go download failed"; FAILED+=("Go"); }
    ln -sf /usr/local/go/bin/go  /usr/local/bin/go  2>/dev/null || true
    ln -sf /usr/local/go/bin/gofmt /usr/local/bin/gofmt 2>/dev/null || true
    export PATH="$PATH:/usr/local/go/bin"
    command -v go &>/dev/null && success "Go ${GO_VERSION} installed" || { error "Go install failed"; FAILED+=("Go"); }
fi
export PATH="$PATH:/usr/local/go/bin:/usr/local/go/bin"

# ── 4. Network reconnaissance ───────────────────────────────────────────────
header "Network Reconnaissance"

# nmap
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "nmap" nmap
else
    install_pkg "nmap" nmap
fi

# masscan
if ! command -v masscan &>/dev/null; then
    info "  Building masscan from source..."
    if [[ "$PKG_MGR" == "dnf" ]]; then
        dnf install -y --quiet libpcap-devel &>/dev/null || true
    else
        apt-get install -y -qq libpcap-dev &>/dev/null || true
    fi
    git clone --depth 1 https://github.com/robertdavidgraham/masscan /tmp/masscan &>/dev/null && \
    make -C /tmp/masscan -j"$(nproc)" &>/dev/null && \
    cp /tmp/masscan/bin/masscan /usr/local/bin/ && \
    rm -rf /tmp/masscan && success "masscan" || { warn "masscan build failed"; FAILED+=("masscan"); }
else
    success "masscan (already installed)"
fi

# rustscan
if ! command -v rustscan &>/dev/null; then
    if command -v cargo &>/dev/null; then
        cargo install rustscan &>/dev/null 2>&1 && success "rustscan" || { warn "rustscan cargo install failed"; FAILED+=("rustscan"); }
    else
        # Try pre-built binary
        RSCAN_URL="https://github.com/RustScan/RustScan/releases/latest/download/rustscan_amd64"
        install_binary "rustscan" "$RSCAN_URL"
    fi
else
    success "rustscan (already installed)"
fi

# ── 5. Subdomain enumeration ─────────────────────────────────────────────────
header "Subdomain Enumeration"

install_go_tool "subfinder"  "github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest"
install_go_tool "amass"      "github.com/owasp-amass/amass/v4/...@latest"
install_go_tool "assetfinder" "github.com/tomnomnom/assetfinder@latest"
install_go_tool "dnsx"       "github.com/projectdiscovery/dnsx/cmd/dnsx@latest"
install_go_tool "shuffledns" "github.com/projectdiscovery/shuffledns/cmd/shuffledns@latest"

# sublist3r
if ! command -v sublist3r &>/dev/null; then
    pip3 install sublist3r -q 2>/dev/null && \
    success "sublist3r" || { warn "sublist3r pip install failed"; FAILED+=("sublist3r"); }
else
    success "sublist3r (already installed)"
fi

# ── 6. HTTP probing & fingerprinting ─────────────────────────────────────────
header "HTTP Probing & Fingerprinting"

install_go_tool "httpx"    "github.com/projectdiscovery/httpx/cmd/httpx@latest"
install_go_tool "katana"   "github.com/projectdiscovery/katana/cmd/katana@latest"
install_go_tool "waybackurls" "github.com/tomnomnom/waybackurls@latest"
install_go_tool "gau"      "github.com/lc/gau/v2/cmd/gau@latest"

# whatweb
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "whatweb" whatweb
else
    install_pkg "whatweb" whatweb
fi

# wafw00f
pip3 install wafw00f -q 2>/dev/null && success "wafw00f" || { warn "wafw00f install failed"; FAILED+=("wafw00f"); }

# nikto
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "nikto" nikto
else
    install_pkg "nikto" nikto
fi

# ── 7. Web fuzzing & directory brute-force ───────────────────────────────────
header "Web Fuzzing"

install_go_tool "ffuf"        "github.com/ffuf/ffuf/v2@latest"
install_go_tool "gobuster"    "github.com/OJ/gobuster/v3@latest"
install_go_tool "feroxbuster" "github.com/epi052/feroxbuster@latest" "feroxbuster"

# feroxbuster via cargo if go failed
if ! command -v feroxbuster &>/dev/null; then
    command -v cargo &>/dev/null && \
    cargo install feroxbuster &>/dev/null 2>&1 && success "feroxbuster (cargo)" || \
    { warn "feroxbuster install failed"; FAILED+=("feroxbuster"); }
fi

# dirsearch
pip3 install dirsearch -q 2>/dev/null && success "dirsearch" || \
    { warn "dirsearch pip install failed"; FAILED+=("dirsearch"); }

# seclists (wordlists)
if [[ ! -d /usr/share/seclists ]]; then
    info "  Installing SecLists wordlists..."
    if [[ "$PKG_MGR" == "dnf" ]]; then
        install_pkg "seclists" seclists
    else
        install_pkg "seclists" seclists
    fi
fi
[[ -d /usr/share/seclists ]] && success "seclists" || \
    { warn "SecLists not in /usr/share/seclists — downloading...";
      git clone --depth 1 https://github.com/danielmiessler/SecLists /usr/share/seclists &>/dev/null && \
      success "seclists (git)" || { warn "SecLists download failed"; FAILED+=("seclists"); }; }

# ── 8. Web vulnerability scanning ────────────────────────────────────────────
header "Web Vulnerability Scanning"

# sqlmap
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "sqlmap" sqlmap
else
    install_pkg "sqlmap" sqlmap
fi

# nuclei
install_go_tool "nuclei" "github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest"
# Update nuclei templates
command -v nuclei &>/dev/null && nuclei -update-templates -silent 2>/dev/null && success "nuclei templates updated" || true

# dalfox (XSS)
install_go_tool "dalfox" "github.com/hahwul/dalfox/v2@latest"

# wpscan (WordPress)
if ! command -v wpscan &>/dev/null; then
    if command -v gem &>/dev/null; then
        gem install wpscan -q 2>/dev/null && success "wpscan" || { warn "wpscan gem install failed"; FAILED+=("wpscan"); }
    else
        install_pkg "ruby + wpscan" ruby ruby-devel && gem install wpscan -q 2>/dev/null && success "wpscan" || \
            { warn "wpscan install failed"; FAILED+=("wpscan"); }
    fi
else
    success "wpscan (already installed)"
fi

# ── 9. OSINT tools ───────────────────────────────────────────────────────────
header "OSINT"

# theHarvester
pip3 install theHarvester -q 2>/dev/null && success "theHarvester" || \
    { warn "theHarvester pip install failed"; FAILED+=("theHarvester"); }

# shodan CLI
pip3 install shodan -q 2>/dev/null && success "shodan CLI" || \
    { warn "shodan CLI install failed"; FAILED+=("shodan"); }

# recon-ng
pip3 install recon-ng -q 2>/dev/null && success "recon-ng" || \
    { warn "recon-ng install failed"; FAILED+=("recon-ng"); }

install_go_tool "github-subdomains" "github.com/gwen001/github-subdomains@latest"
install_go_tool "hakrawler"         "github.com/hakluke/hakrawler@latest"

# ── 10. Active Directory / Windows tools ─────────────────────────────────────
header "Active Directory & Windows"

# impacket
pip3 install impacket -q 2>/dev/null && success "impacket" || \
    { warn "impacket install failed"; FAILED+=("impacket"); }

# crackmapexec → netexec (successor)
pip3 install netexec -q 2>/dev/null && success "netexec (cme successor)" || \
    pip3 install crackmapexec -q 2>/dev/null && success "crackmapexec" || \
    { warn "crackmapexec/netexec install failed"; FAILED+=("crackmapexec"); }

# kerbrute
install_go_tool "kerbrute" "github.com/ropnop/kerbrute@latest"

# enum4linux-ng
pip3 install enum4linux-ng -q 2>/dev/null && success "enum4linux-ng" || \
    { warn "enum4linux-ng install failed"; FAILED+=("enum4linux-ng"); }

# ldapdomaindump
pip3 install ldapdomaindump -q 2>/dev/null && success "ldapdomaindump" || \
    { warn "ldapdomaindump install failed"; FAILED+=("ldapdomaindump"); }

# bloodhound-python (collection)
pip3 install bloodhound -q 2>/dev/null && success "bloodhound-python" || \
    { warn "bloodhound-python install failed"; FAILED+=("bloodhound-python"); }

# evil-winrm
if ! command -v evil-winrm &>/dev/null; then
    command -v gem &>/dev/null && \
    gem install evil-winrm -q 2>/dev/null && success "evil-winrm" || \
    { warn "evil-winrm install failed"; FAILED+=("evil-winrm"); }
else
    success "evil-winrm (already installed)"
fi

# ── 11. Exploitation frameworks ───────────────────────────────────────────────
header "Exploitation Frameworks"

# metasploit
if ! command -v msfconsole &>/dev/null; then
    if [[ "$PKG_MGR" == "dnf" ]]; then
        # Fedora: use the official installer
        info "  Installing Metasploit Framework via official installer..."
        curl -sSfL https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb \
            -o /tmp/msfinstall 2>/dev/null && \
        chmod +x /tmp/msfinstall && /tmp/msfinstall &>/dev/null && \
        success "metasploit" || { warn "Metasploit install failed — install manually from https://metasploit.com/download"; FAILED+=("metasploit"); }
    else
        curl -sSfL https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb \
            -o /tmp/msfinstall 2>/dev/null && \
        chmod +x /tmp/msfinstall && /tmp/msfinstall &>/dev/null && \
        success "metasploit" || { warn "Metasploit install failed"; FAILED+=("metasploit"); }
    fi
else
    success "metasploit (already installed)"
fi

# pymetasploit3 (for Mythos Engine RPC)
pip3 install pymetasploit3 -q 2>/dev/null && success "pymetasploit3" || \
    { warn "pymetasploit3 install failed"; FAILED+=("pymetasploit3"); }

# ── 12. Credential & post-exploitation tools ──────────────────────────────────
header "Credential & Post-Exploitation"

# hydra
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "hydra" hydra
else
    install_pkg "hydra" hydra
fi

# john
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "john" john
else
    install_pkg "john" john
fi

# hashcat
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "hashcat" hashcat
else
    install_pkg "hashcat" hashcat
fi

# chisel (tunneling)
install_go_tool "chisel" "github.com/jpillora/chisel@latest"

# ligolo-ng (tunneling)
install_go_tool "ligolo-ng" "github.com/nicocha30/ligolo-ng/cmd/proxy@latest" "proxy"

# ── 13. Cloud tools ────────────────────────────────────────────────────────────
header "Cloud Security"

# awscli
if ! command -v aws &>/dev/null; then
    info "  Installing AWS CLI v2..."
    ARCH=$(uname -m)
    curl -sSfL "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o /tmp/awscliv2.zip 2>/dev/null && \
    unzip -q /tmp/awscliv2.zip -d /tmp/ && \
    /tmp/aws/install -i /usr/local/aws-cli -b /usr/local/bin &>/dev/null && \
    rm -rf /tmp/awscliv2.zip /tmp/aws && success "aws-cli" || \
    { warn "AWS CLI install failed"; FAILED+=("aws-cli"); }
else
    success "aws-cli (already installed)"
fi

# azure cli
if ! command -v az &>/dev/null; then
    if [[ "$PKG_MGR" == "dnf" ]]; then
        rpm --import https://packages.microsoft.com/keys/microsoft.asc 2>/dev/null || true
        dnf install -y --quiet azure-cli 2>/dev/null && success "azure-cli" || \
        { warn "azure-cli install failed"; FAILED+=("azure-cli"); }
    else
        curl -sL https://aka.ms/InstallAzureCLIDeb | bash &>/dev/null && success "azure-cli" || \
        { warn "azure-cli install failed"; FAILED+=("azure-cli"); }
    fi
else
    success "azure-cli (already installed)"
fi

# scout suite (cloud auditing)
pip3 install scoutsuite -q 2>/dev/null && success "scoutsuite" || \
    { warn "scoutsuite install failed"; FAILED+=("scoutsuite"); }

# pacu (AWS exploitation)
pip3 install pacu -q 2>/dev/null && success "pacu" || \
    { warn "pacu install failed"; FAILED+=("pacu"); }

# prowler (cloud security)
pip3 install prowler -q 2>/dev/null && success "prowler" || \
    { warn "prowler install failed"; FAILED+=("prowler"); }

# ── 14. HTTP utilities ─────────────────────────────────────────────────────────
header "HTTP Utilities"

# curl + wget already in build deps
command -v curl &>/dev/null && success "curl" || install_pkg "curl" curl
command -v wget &>/dev/null && success "wget" || install_pkg "wget" wget

# jq
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "jq" jq
else
    install_pkg "jq" jq
fi

# proxychains
if [[ "$PKG_MGR" == "dnf" ]]; then
    install_pkg "proxychains" proxychains-ng
else
    install_pkg "proxychains" proxychains4
fi

# ── 15. Mythos Engine Python deps ──────────────────────────────────────────────
header "Mythos Engine Python Dependencies"

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${PROJ_DIR}/mythos_engine/requirements.txt" ]]; then
    pip3 install -r "${PROJ_DIR}/mythos_engine/requirements.txt" -q 2>/dev/null && \
    success "mythos_engine/requirements.txt" || warn "Some Python deps failed"
fi
if [[ -f "${PROJ_DIR}/requirements.txt" ]]; then
    pip3 install -r "${PROJ_DIR}/requirements.txt" -q 2>/dev/null && \
    success "requirements.txt" || warn "Some Python deps failed"
fi

# ── 16. Verification ───────────────────────────────────────────────────────────
header "Installation Verification"

TOOLS=(
    nmap masscan httpx subfinder amass nuclei ffuf gobuster
    sqlmap nikto whatweb curl wget jq hydra hashcat chisel
    dalfox kerbrute
)

ALL_OK=true
for t in "${TOOLS[@]}"; do
    if command -v "$t" &>/dev/null; then
        printf "  ${GREEN}✓${RESET} %-20s %s\n" "$t" "$(command -v "$t")"
    else
        printf "  ${RED}✗${RESET} %-20s NOT FOUND\n" "$t"
        ALL_OK=false
    fi
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    warn "The following tools failed to install and need manual attention:"
    for f in "${FAILED[@]}"; do
        echo -e "  ${RED}•${RESET} $f"
    done
    echo ""
    warn "Common fixes:"
    echo "  • masscan/rustscan:   cargo install rustscan  OR  build from source"
    echo "  • amass:              go install github.com/owasp-amass/amass/v4/...@latest"
    echo "  • metasploit:         https://metasploit.com/download"
    echo "  • bloodhound:         https://github.com/BloodHoundAD/BloodHound"
    echo "  • wpscan:             gem install wpscan"
    echo ""
fi

if $ALL_OK; then
    success "All core tools installed successfully!"
else
    warn "Some tools could not be installed. See above."
fi

echo -e "\n${BOLD}Mythos Engine is ready. Run with:${RESET}"
echo "  cd mythos_engine && python3 run.py --target <host> --bug-bounty --scope ../engagements/coupang-tw/scope.json --rag --execute"
