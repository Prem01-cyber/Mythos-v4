<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Discord][discord-shield]][discord-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">PentestGPT</h3>

  <p align="center">
    AI-Powered Autonomous Penetration Testing Agent
    <br />
    <strong>Published at USENIX Security 2024</strong>
    <br />
    <br />
    <a href="https://pentestgpt.com"><strong>Official Website: pentestgpt.com »</strong></a>
    <br />
    <br />
    <a href="https://www.usenix.org/conference/usenixsecurity24/presentation/deng">Research Paper</a>
    ·
    <a href="https://github.com/GreyDGL/PentestGPT/issues">Report Bug</a>
    ·
    <a href="https://github.com/GreyDGL/PentestGPT/issues">Request Feature</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
<a href="https://trendshift.io/repositories/3770" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3770" alt="GreyDGL%2FPentestGPT | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

---

## Demo

### Installation
[![Installation Demo](https://asciinema.org/a/761661.svg)](https://asciinema.org/a/761661)

[Watch on YouTube](https://www.youtube.com/watch?v=RUNmoXqBwVg)

### PentestGPT in Action
[![PentestGPT Demo](https://asciinema.org/a/761663.svg)](https://asciinema.org/a/761663)

[Watch on YouTube](https://www.youtube.com/watch?v=cWi3Yb7RmZA)

---

## What's New in v1.0 (Agentic Upgrade)

- **Autonomous Agent** - Agentic pipeline for intelligent, autonomous penetration testing
- **Session Persistence** - Save and resume penetration testing sessions
- **Docker-First** - Isolated, reproducible environment with security tools pre-installed

> **In Progress**: Multi-model support for OpenAI, Gemini, and other LLM providers

---

## Features

- **AI-Powered Challenge Solver** - Leverages LLM advanced reasoning to perform penetration testing and CTFs
- **Live Walkthrough** - Tracks steps in real-time as the agent works through challenges
- **Multi-Category Support** - Web, Crypto, Reversing, Forensics, PWN, Privilege Escalation
- **Real-Time Feedback** - Watch the AI work with live activity updates
- **Extensible Architecture** - Clean, modular design ready for future enhancements

---

## Quick Start

### Prerequisites

- **Docker** (required) - [Install Docker](https://docs.docker.com/get-docker/)
- **LLM Provider** (choose one):
  - Anthropic API Key from [console.anthropic.com](https://console.anthropic.com/)
  - Claude OAuth Login (requires Claude subscription)
  - OpenRouter for alternative models at [openrouter.ai](https://openrouter.ai/keys)
  - [Tutorial: Using Local Models with Claude Code](https://docs.google.com/document/d/1ixK7x-wlr5t5TYZJdfm75UME5KnPCpS46boLkUXKg1w/edit?usp=sharing)


### Installation

```bash
# Clone and build
git clone --recurse-submodules https://github.com/GreyDGL/PentestGPT.git
cd PentestGPT
make install

# Configure authentication (first time only)
make config

# Connect to container
make connect
```

> **Note**: The `--recurse-submodules` flag downloads the benchmark suite. If you already cloned without it, run: `git submodule update --init --recursive`

### Try a Benchmark

```bash
cd benchmark/standalone-xbow-benchmark-runner
python3 run_benchmarks.py --range 1-1 --pattern-flag
```

See [Benchmark Documentation](benchmark/README.md) for detailed usage.

### Commands Reference

| Command | Description |
|---------|-------------|
| `make install` | Build the Docker image |
| `make config` | Configure API key (first-time setup) |
| `make connect` | Connect to container (main entry point) |
| `make stop` | Stop container (config persists) |
| `make clean-docker` | Remove everything including config |


---

## Usage

```bash
# Interactive TUI mode (default)
pentestgpt --target 10.10.11.234

# Non-interactive mode
pentestgpt --target 10.10.11.100 --non-interactive

# With challenge context
pentestgpt --target 10.10.11.50 --instruction "WordPress site, focus on plugin vulnerabilities"
```

**Keyboard Shortcuts:** `F1` Help | `Ctrl+P` Pause/Resume | `Ctrl+Q` Quit

---

## Using Local LLMs

PentestGPT supports routing requests to local LLM servers (LM Studio, Ollama, text-generation-webui, etc.) running on your host machine.

### Prerequisites

- Local LLM server with an OpenAI-compatible API endpoint
  - **LM Studio**: Enable server mode (default port 1234)
  - **Ollama**: Run `ollama serve` (default port 11434)

### Setup

```bash
# Configure PentestGPT for local LLM
make config
# Select option 4: Local LLM

# Start your local LLM server on the host machine
# Then connect to the container
make connect
```

### Customizing Models

Edit `scripts/ccr-config-template.json` to customize:

- **`localLLM.api_base_url`**: Your LLM server URL (default: `host.docker.internal:1234`)
- **`localLLM.models`**: Available model names on your server
- **Router section**: Which models handle which operations

| Route | Purpose | Default Model |
|-------|---------|---------------|
| `default` | General tasks | openai/gpt-oss-20b |
| `background` | Background operations | openai/gpt-oss-20b |
| `think` | Reasoning-heavy tasks | qwen/qwen3-coder-30b |
| `longContext` | Large context handling | qwen/qwen3-coder-30b |
| `webSearch` | Web search operations | openai/gpt-oss-20b |

### Troubleshooting

- **Connection refused**: Ensure your LLM server is running and listening on the configured port
- **Docker networking**: Use `host.docker.internal` (not `localhost`) to access host services from Docker
- **Check CCR logs**: Inside the container, run `cat /tmp/ccr.log`

---

## Telemetry

PentestGPT collects anonymous usage data to help improve the tool. This data is sent to our [Langfuse](https://langfuse.com) project and includes:
- Session metadata (target type, duration, completion status)
- Tool execution patterns (which tools are used, not the actual commands)
- Flag detection events (that a flag was found, not the flag content)

**No sensitive data is collected** - command outputs, credentials, or actual flag values are never transmitted.

### Opting Out

```bash
# Via command line flag
pentestgpt --target 10.10.11.234 --no-telemetry

# Via environment variable
export LANGFUSE_ENABLED=false
```

---

## Benchmarks

PentestGPT includes 104 XBOW validation benchmarks for comprehensive testing and evaluation.

```bash
cd benchmark/standalone-xbow-benchmark-runner

python3 run_benchmarks.py --range 1-10 --pattern-flag   # Run benchmarks 1-10
python3 run_benchmarks.py --all --pattern-flag          # Run all 104 benchmarks
python3 run_benchmarks.py --retry-failed                # Retry failed benchmarks
python3 run_benchmarks.py --dry-run --range 1-5         # Preview without executing
```

### Performance Highlights

PentestGPT achieved an **86.5% success rate** (90/104 benchmarks) on the XBOW validation suite:

- **Cost**: Average $1.11, Median $0.42 per successful benchmark
- **Time**: Average 6.1 minutes, Median 3.3 minutes per successful benchmark
- **Success rates by difficulty**:
  - Level 1: 91.1%
  - Level 2: 74.5%
  - Level 3: 62.5%

For detailed benchmark results, analysis, and automated testing instructions, see the **[Benchmark Documentation](benchmark/README.md)**.

---

## Legacy Version

The previous multi-LLM version (v0.15) supporting OpenAI, Gemini, Deepseek, and Ollama is archived in [`legacy/`](legacy/):

```bash
cd legacy && pip install -e . && pentestgpt --reasoning gpt-4o
```

---

## Citation

If you use PentestGPT in your research, please cite our paper:

```bibtex
@inproceedings{299699,
  author = {Gelei Deng and Yi Liu and Víctor Mayoral-Vilches and Peng Liu and Yuekang Li and Yuan Xu and Tianwei Zhang and Yang Liu and Martin Pinzger and Stefan Rass},
  title = {{PentestGPT}: Evaluating and Harnessing Large Language Models for Automated Penetration Testing},
  booktitle = {33rd USENIX Security Symposium (USENIX Security 24)},
  year = {2024},
  isbn = {978-1-939133-44-1},
  address = {Philadelphia, PA},
  pages = {847--864},
  url = {https://www.usenix.org/conference/usenixsecurity24/presentation/deng},
  publisher = {USENIX Association},
  month = aug
}
```

---

## License

Distributed under the MIT License. See `LICENSE.md` for more information.

**Disclaimer**: This tool is for educational purposes and authorized security testing only. The authors do not condone any illegal use. Use at your own risk.

---

## Acknowledgments

- Research supported by [Quantstamp](https://www.quantstamp.com/) and [NTU Singapore](https://www.ntu.edu.sg/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/GreyDGL/PentestGPT.svg?style=for-the-badge
[contributors-url]: https://github.com/GreyDGL/PentestGPT/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/GreyDGL/PentestGPT.svg?style=for-the-badge
[forks-url]: https://github.com/GreyDGL/PentestGPT/network/members
[stars-shield]: https://img.shields.io/github/stars/GreyDGL/PentestGPT.svg?style=for-the-badge
[stars-url]: https://github.com/GreyDGL/PentestGPT/stargazers
[issues-shield]: https://img.shields.io/github/issues/GreyDGL/PentestGPT.svg?style=for-the-badge
[issues-url]: https://github.com/GreyDGL/PentestGPT/issues
[license-shield]: https://img.shields.io/github/license/GreyDGL/PentestGPT.svg?style=for-the-badge
[license-url]: https://github.com/GreyDGL/PentestGPT/blob/master/LICENSE.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/gelei-deng-225a10112/
[linkedin-url2]: https://www.linkedin.com/in/vmayoral/
[discord-shield]: https://dcbadge.vercel.app/api/server/eC34CEfEkK
[discord-url]: https://discord.gg/eC34CEfEkK