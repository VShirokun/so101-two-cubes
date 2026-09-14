#!/usr/bin/env bash
# Постоянный «адрес» без белого IP и без аккаунтов: быстрый туннель Cloudflare под
# присмотром. При каждом (пере)запуске новый адрес публикуется туда, где его можно
# прочитать с телефона: файл real/teleop/TUNNEL_URL.md в приватном репозитории
# Office-Rover на GitHub (через API, токен в /srv/data/vshirokun/.secrets/github_token),
# и локально /tmp/roboom_tunnel_url.txt. Ключ доступа берётся из /tmp/roboom_teleop.key.
#   tmux new -d -s tunnel "bash real/teleop/tunnel_supervisor.sh"
set -u
CF=/srv/data/vshirokun/bin/cloudflared
REPO=VShirokun/Office-Rover
TOKEN_FILE=/srv/data/vshirokun/.secrets/github_token
publish() {   # $1 = url
  local key url_full body sha
  key=$(cat /tmp/roboom_teleop.key 2>/dev/null || echo "")
  url_full="$1/?k=$key"
  echo "$url_full" > /tmp/roboom_tunnel_url.txt
  [ -r "$TOKEN_FILE" ] || return 0
  body=$(printf '# Адрес телеоперации\n\nОбновляется автоматически при перезапуске туннеля (%s).\n\n- Джойстик: %s\n- AR (Android): %s/xr?k=%s\n- Наклоны (iPhone): %s/tilt?k=%s\n' "$(date -Is)" "$url_full" "$1" "$key" "$1" "$key" | base64 -w0)
  sha=$(curl -s -m 20 -H "Authorization: Bearer $(cat $TOKEN_FILE)" "https://api.github.com/repos/$REPO/contents/real/teleop/TUNNEL_URL.md" | python3 -c "import json,sys; print(json.load(sys.stdin).get('sha',''))" 2>/dev/null)
  curl -s -m 20 -o /dev/null -w "github: %{http_code}\n" -X PUT -H "Authorization: Bearer $(cat $TOKEN_FILE)" "https://api.github.com/repos/$REPO/contents/real/teleop/TUNNEL_URL.md" \
    -d "{\"message\":\"tunnel: новый адрес телеоперации\",\"content\":\"$body\"$( [ -n "$sha" ] && printf ',"sha":"%s"' "$sha" )}"
}
while true; do
  echo "$(date -Is) старт туннеля"
  $CF tunnel --url http://localhost:8080 --no-autoupdate 2>&1 | while read -r line; do
    echo "$line"
    u=$(echo "$line" | grep -o "https://[a-z0-9-]*\.trycloudflare\.com" | head -1)
    if [ -n "$u" ] && [ "$u" != "$(cat /tmp/roboom_tunnel_last 2>/dev/null)" ]; then echo "$u" > /tmp/roboom_tunnel_last; echo "$(date -Is) адрес: $u"; publish "$u"; fi
  done
  echo "$(date -Is) туннель упал, перезапуск через 5 с"; sleep 5
done
