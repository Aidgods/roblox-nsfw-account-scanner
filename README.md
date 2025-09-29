---

# 🕵️ Roblox NSFW Account Scanner

A lightweight Python-based scanner that flags potentially NSFW Roblox accounts using a custom AI model. Built for moderation, research, or curiosity — not harassment.

---

## ⚠️ Disclaimer

This tool is for **educational and moderation purposes only**.  
Misuse (e.g., harassment, doxxing, or spam) violates Roblox ToS and may be illegal.  
Use responsibly and at your own risk.

---

## 🧠 Accuracy

- ~70% accuracy  
- Custom AI model (not GPT or commercial)  
- False positives/negatives expected — always review results manually

---

## 🛠️ Installation

### Linux (Recommended)
```bash
git clone https://github.com/Aidgods/roblox-nsfw-account-scanner.git
cd roblox-nsfw-account-scanner
pip install -r requirements.txt
```

> ⚠️ Windows users: try running on **WSL** or a **Linux VPS**.

---

## 🚀 Usage

### Option 1: Double-click
Open `run.bat` (Windows) or `run.sh` (Linux)

### Option 2: CLI
```bash
python robloxscanner.py
```

---

## 📝 Customize Targets

Add usernames to `keywords.txt`, one per line:
```
baduser123
another sus_name
xXx_NoobSlayer_xXx
```

---

## 🧩 Requirements

- Python 3.8+  
- See `requirements.txt` for Python packages

---

## 🤝 Contributing

Pull requests welcome — especially for:
- Improving the AI model
- Adding proxy support
- Better error handling

---

Let me know if you want a version with badges, a logo, or a Docker setup.
