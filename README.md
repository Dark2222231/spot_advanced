
# Spot Advanced Bot - Deployment Archive

This archive contains an advanced Telegram bot for multi-timeframe technical and structure analysis (FVG, Order Blocks, Liquidity, Orderbook pressure).

## Contents
- `spot_advanced_bot.py` - main bot script
- `requirements.txt` - Python dependencies
- `systemd/spotbot.service` - example systemd unit for Ubuntu/Debian
- `.env.example` - example environment file
- `README.md` - this file

## Quick setup (Ubuntu 22.04 / Oracle Cloud Always Free recommended)

1. Prepare VM (Ubuntu) and connect via SSH.
2. Install Python and create virtualenv:
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
3. Copy files to `/home/ubuntu/spotbot` (or any folder). Create `.env` file and set TELEGRAM_BOT_TOKEN:
```bash
cp .env.example .env
# edit .env and set TELEGRAM_BOT_TOKEN
```
4. Run manually to test:
```bash
source venv/bin/activate
export TELEGRAM_BOT_TOKEN="YOUR_TOKEN"
python3 spot_advanced_bot.py
```
5. To run as service, copy `systemd/spotbot.service` to `/etc/systemd/system/spotbot.service`, edit paths and user, then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable spotbot
sudo systemctl start spotbot
sudo journalctl -u spotbot -f
```

## Notes & Security
- Replace the token; keep it secret.
- Bot uses public Binance market data endpoints (no API keys required for public data). For orderbook depth and user streams you may choose to add API keys.
- This tool is informational — test strategies on paper and use risk management.

