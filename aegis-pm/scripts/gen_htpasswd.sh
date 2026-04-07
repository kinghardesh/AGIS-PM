#!/usr/bin/env bash
# =============================================================================
#  Aegis PM – scripts/gen_htpasswd.sh
#
#  Generates nginx/.htpasswd from DASHBOARD_HTPASSWD_HASH in your .env file,
#  OR interactively prompts for username + password.
#
#  Usage:
#    # From .env (DASHBOARD_HTPASSWD_HASH must be set)
#    bash scripts/gen_htpasswd.sh
#
#    # Interactive (prompts for username and password)
#    bash scripts/gen_htpasswd.sh --interactive
#
#    # Quick one-liner to generate a hash you can paste into .env:
#    htpasswd -nb admin yourpassword
# =============================================================================

set -euo pipefail

HTPASSWD_FILE="nginx/.htpasswd"
mkdir -p nginx

if [[ "${1:-}" == "--interactive" ]]; then
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  Aegis PM – Dashboard Auth Setup"
    echo "═══════════════════════════════════════════"
    echo ""
    read -rp  "Username (e.g. admin): " USERNAME
    read -rsp "Password: " PASSWORD
    echo ""
    read -rsp "Confirm password: " PASSWORD2
    echo ""

    if [[ "$PASSWORD" != "$PASSWORD2" ]]; then
        echo "❌ Passwords do not match."
        exit 1
    fi

    if ! command -v htpasswd &> /dev/null; then
        echo "htpasswd not found. Installing apache2-utils…"
        apt-get install -y apache2-utils 2>/dev/null || \
        brew install httpd 2>/dev/null || \
        { echo "❌ Please install apache2-utils (Linux) or httpd (macOS) manually."; exit 1; }
    fi

    htpasswd -cbn "$USERNAME" "$PASSWORD" > "$HTPASSWD_FILE"
    echo ""
    echo "✅ Created $HTPASSWD_FILE"
    echo ""
    echo "Add this line to your .env:"
    echo "  DASHBOARD_HTPASSWD_HASH=$(cat "$HTPASSWD_FILE")"

elif [[ -f ".env" ]]; then
    # Load from .env
    HASH=$(grep '^DASHBOARD_HTPASSWD_HASH=' .env | cut -d'=' -f2- | tr -d '"' | tr -d "'")

    if [[ -z "$HASH" ]] || [[ "$HASH" == "admin:\$apr1\$..."* ]]; then
        echo ""
        echo "⚠️  DASHBOARD_HTPASSWD_HASH not set in .env"
        echo ""
        echo "Run interactively to generate one:"
        echo "  bash scripts/gen_htpasswd.sh --interactive"
        echo ""
        echo "Or generate a hash manually and add to .env:"
        echo "  htpasswd -nb admin yourpassword"
        exit 1
    fi

    echo "$HASH" > "$HTPASSWD_FILE"
    echo "✅ Created $HTPASSWD_FILE from .env"
else
    echo "❌ No .env file found and --interactive not specified."
    echo "Run: bash scripts/gen_htpasswd.sh --interactive"
    exit 1
fi

chmod 640 "$HTPASSWD_FILE"
echo ""
echo "Next: mount this file in docker-compose.yml:"
echo "  volumes:"
echo "    - ./nginx/.htpasswd:/etc/nginx/auth/.htpasswd:ro"
echo ""
echo "Or rebuild the nginx container:"
echo "  docker compose up --build frontend"
