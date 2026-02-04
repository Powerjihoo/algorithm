base_dir="/home/Model_Data"

# 모든 직속 하위 디렉토리 및 파일 찾기 및 이름 변경
find "$base_dir" -maxdepth 2 -mindepth 2 -exec bash -c '
  for item; do
    new_item="${item%/*}/$(tr "[:upper:]" "[:lower:]" <<< "${item##*/}")"
    if [ "$item" != "$new_item" ]; then
      mv "$item" "$new_item"
    fi
  done
' bash {} +