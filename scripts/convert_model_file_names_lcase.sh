#!/bin/bash

# 기본 디렉토리 경로 설정
base_dir="/home/Model_Data"

# Model_Data 직속 하위 디렉토리 찾기 및 이름 변경
for dir in "$base_dir"/*; do
  if [ -d "$dir" ]; then
    new_dir="${dir%/*}/$(tr "[:upper:]" "[:lower:]" <<< "${dir##*/}")"
    if [ "$dir" != "$new_dir" ]; then
      mv "$dir" "$new_dir"
    fi
  fi
done

# 모든 직속 하위 디렉토리 및 파일 찾기 및 이름 변경
find "$base_dir" -maxdepth 2 -mindepth 2 -exec bash -c '
  for item; do
    new_item="${item%/*}/$(tr "[:upper:]" "[:lower:]" <<< "${item##*/}")"
    if [ "$item" != "$new_item" ]; then
      mv "$item" "$new_item"
    fi
  done
' bash {} +