import os
import re
from pathlib import Path


def parse_git_patch(patch_text):
    """解析 Git 补丁内容，返回文件路径和修改列表"""
    files = []
    current_file = None
    lines = patch_text.splitlines()

    for line in lines:
        file_match = re.match(r'^diff --git a/(.+) b/\1', line)
        if file_match:
            if current_file:
                files.append(current_file)
            current_file = {
                'path': file_match.group(1),
                'hunks': []
            }
            continue

        hunk_match = re.match(r'^@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@', line)
        if hunk_match and current_file:
            old_start, old_len = int(hunk_match.group(1)), int(hunk_match.group(2) or 1)
            new_start, new_len = int(hunk_match.group(3)), int(hunk_match.group(4) or 1)
            current_hunk = {
                'old_start': old_start,
                'old_len': old_len,
                'new_start': new_start,
                'new_len': new_len,
                'lines': []
            }
            current_file['hunks'].append(current_hunk)
            continue

        if current_file and current_file['hunks']:
            current_hunk = current_file['hunks'][-1]
            if line.startswith('\\ No newline at end of file'):
                continue
            current_hunk['lines'].append(line)

    if current_file:
        files.append(current_file)
    return files


def apply_hunk_to_file(file_path, hunk, content):
    """将 hunk 应用到文件内容"""
    lines = content.splitlines()
    hunk_lines = hunk['lines']

    # 提取上下文和修改内容
    context = []
    additions = []
    deletions = []

    for line in hunk_lines:
        if line.startswith(' ') or not line.startswith(('+', '-')):
            context.append(line[1:] if line else '')
        elif line.startswith('+'):
            additions.append(line[1:])
        elif line.startswith('-'):
            deletions.append(line[1:])

    # 尝试匹配上下文
    matched = False
    for i in range(max(0, hunk['old_start'] - 5), min(len(lines), hunk['old_start'] + 5)):
        match = True
        for j, ctx_line in enumerate(context):
            if i + j >= len(lines):
                match = False
                break
            target_line = lines[i + j].strip()
            if ctx_line.strip() != target_line:
                match = False
                break
        if match:
            del lines[i:i + len(deletions)]
            for k, add_line in enumerate(additions):
                lines.insert(i + k, add_line)
            matched = True
            break

    if not matched:
        print(f"[!] 无法匹配上下文，跳过替换: {file_path}")
        return content

    return '\n'.join(lines)


def apply_patch_manually(patch_text, base_dir='.', backup_dir='copy_files'):
    """主函数：解析补丁并替换文件内容"""
    files = parse_git_patch(patch_text)
    backup_dir = Path(backup_dir).absolute()

    # 创建备份目录（如果不存在）
    backup_dir.mkdir(parents=True, exist_ok=True)

    for file_info in files:
        file_path = Path(base_dir) / file_info['path']
        if not file_path.exists():
            print(f"[!] 文件不存在: {file_path}")
            continue

        print(f"\n[+] 正在处理文件: {file_path}")

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 构造备份路径（保留原文件结构）
        relative_path = file_path.relative_to(base_dir)
        backup_path = backup_dir / relative_path.with_suffix(file_path.suffix + '.bak')

        # 创建备份目录结构
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # 备份文件
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✓] 已备份原文件到: {backup_path}")

        # 应用每个 hunk
        for hunk in file_info['hunks']:
            content = apply_hunk_to_file(file_path, hunk, content)

        # 写入新内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✓] 已更新文件: {file_path}")


if __name__ == "__main__":
    import sys

    # 从命令行参数或 stdin 读取补丁内容
    if len(sys.argv) > 1:
        patch_path = sys.argv[1]
        with open(patch_path, 'r', encoding='utf-8') as f:
            patch_text = f.read()
    else:
        print("请粘贴 Git 补丁内容（Ctrl+Z 或 Ctrl+D 结束输入）:")
        patch_text = sys.stdin.read()

    apply_patch_manually(patch_text)