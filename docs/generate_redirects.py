import yaml
import os
from pathlib import Path
import argparse

# --- 設定 ---
# <meta> タグのテンプレート。URLはサイトルートからの絶対パスを想定
REDIRECT_TEMPLATE = '<meta http-equiv="Refresh" content="0; url={}" />'
# デフォルトの出力YAMLファイル名
DEFAULT_OUTPUT_YAML = "generated_redirects.yaml"
# デフォルトのリダイレクトHTML出力ディレクトリ
DEFAULT_REDIRECT_DIR = Path("redirect")
# HTMLに変換されると想定するソースファイルの拡張子
DEFAULT_SOURCE_EXTENSIONS = {".md", ".ipynb"}
# ルール生成時に無視するディレクトリ名（この名前がパスに含まれる場合）
EXCLUDE_DIRS = {"assets", "_static", "_build", ".ipynb_checkpoints"}
# ルール生成時に無視するファイル名
EXCLUDE_FILES = {"_config.yml", "_toc.yml", ".readthedocs.yaml"}

# --- 関数: ディレクトリ走査とルール生成 ---
def generate_redirect_rules(source_dir, lang_prefix):
    """指定されたディレクトリを走査し、リダイレクトルールのリストを生成する"""
    source_path = Path(source_dir)
    rules = []
    print(f"ディレクトリ '{source_path}' を走査中...")
    for item in source_path.rglob("*"): # 再帰的に検索
        # 除外ディレクトリ/ファイルのチェック
        if any(part in EXCLUDE_DIRS for part in item.parts):
            continue
        if item.name in EXCLUDE_FILES:
            continue

        # 対象拡張子のファイルかチェック
        if item.is_file() and item.suffix in DEFAULT_SOURCE_EXTENSIONS:
            # ソースディレクトリからの相対パスを取得
            relative_path = item.relative_to(source_path)
            # .html に変換したパスを生成 (拡張子を変更)
            html_path = relative_path.with_suffix(".html")

            # リダイレクト元 (from): ルートからのパス (先頭に / を追加)
            from_path = f"/{html_path}"
            # リダイレクト先 (to): 言語プレフィックス付きのルートからのパス
            to_path = f"/{lang_prefix}/{html_path}"

            # index.html の場合、親ディレクトリのパスを使うか検討
            # 例: /user_guide/index.html -> from: /user_guide/, to: /en/user_guide/
            # Jupyter Book の出力仕様に合わせて調整が必要な場合あり
            # if html_path.name == "index.html":
            #     # 親ディレクトリがルートでない場合のみ末尾に / をつける
            #     parent_dir_str = str(html_path.parent)
            #     if parent_dir_str != ".": # ルートディレクトリ直下のindex.htmlでない
            #         from_path = f"/{parent_dir_str}/"
            #         to_path = f"/{lang_prefix}/{parent_dir_str}/"
            #     else: # ルートのindex.html
            #         from_path = "/"
            #         to_path = f"/{lang_prefix}/" # or f"/{lang_prefix}/index.html"

            rules.append({"from": from_path, "to": to_path})
            # print(f"  ルール追加: {from_path} -> {to_path}")

    print(f"{len(rules)} 件のリダイレクトルール候補を生成しました。")
    return rules

# --- 関数: 設定ファイル書き込み ---
def write_yaml_config(rules, output_file):
    """リダイレクトルールのリストをYAMLファイルに書き込む"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True) # 必要なら親ディレクトリ作成
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # YAML形式で書き出し (リストの各要素を独立したドキュメントとして表示)
            yaml.dump(rules, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"リダイレクト設定を '{output_path}' に書き込みました。")
    except Exception as e:
        print(f"エラー: 設定ファイル '{output_path}' の書き込みに失敗しました: {e}")

# --- 関数: リダイレクトHTML生成 ---
def generate_redirect_files(config_file, redirect_dir):
    """YAML設定ファイルを読み込み、リダイレクトHTMLファイルを生成する"""
    config_path = Path(config_file)
    redirect_path = Path(redirect_dir)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            rules = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"エラー: 設定ファイル '{config_path}' が見つかりません。HTML生成をスキップします。")
        return
    except yaml.YAMLError as e:
        print(f"エラー: 設定ファイル '{config_path}' の読み込みエラー: {e}. HTML生成をスキップします。")
        return
    except Exception as e:
        print(f"エラー: 設定ファイル '{config_path}' の読み込み中に予期せぬエラー: {e}. HTML生成をスキップします。")
        return


    if not isinstance(rules, list):
        print(f"エラー: 設定ファイル '{config_path}' の形式がリストではありません。HTML生成をスキップします。")
        return
    if not rules:
        print("設定ファイルにリダイレクトルールが見つかりません。HTML生成をスキップします。")
        return

    print(f"'{redirect_path}' ディレクトリへのリダイレクトHTMLファイル生成を開始...")
    redirect_path.mkdir(exist_ok=True) # 出力先ディレクトリがなければ作成
    generated_count = 0

    for rule in rules:
        if not isinstance(rule, dict):
            print(f"警告: 不正なルール形式です (dictではありません): {rule}")
            continue
        from_path = rule.get("from")
        to_path = rule.get("to")

        if not from_path or not to_path or not from_path.startswith('/'):
            print(f"警告: 不正なルールです ('from'/'to'がないか、'from'が'/'で始まっていません): {rule}")
            continue

        # from_path の先頭の / を削除して、出力先のファイルパスを決定
        # 例: /user_guide/page.html -> redirect/user_guide/page.html
        output_file_path = redirect_path / from_path.lstrip('/')

        # 出力先の親ディレクトリが存在しなければ作成
        try:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
             print(f"エラー: ディレクトリ '{output_file_path.parent}' の作成に失敗: {e}. このルールの処理をスキップします。")
             continue

        # HTMLファイルを作成
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                # リダイレクト先URLはサイトルートからの絶対パス (例: /en/user_guide/page.html)
                f.write(REDIRECT_TEMPLATE.format(to_path))
            # print(f"  生成: {output_file_path} -> {to_path}")
            generated_count += 1
        except Exception as e:
            print(f"エラー: ファイル '{output_file_path}' の書き込みに失敗: {e}")


    print(f"{generated_count} 件のリダイレクトHTMLファイルを '{redirect_path}' に生成しました。")


# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="指定された言語ディレクトリからリダイレクトルールを生成し、リダイレクトHTMLファイルを作成します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # デフォルト値をヘルプに表示
    )
    parser.add_argument("source_dir", help="リダイレクトルール生成元の言語ディレクトリ (例: docs/en)")
    parser.add_argument("lang_prefix", help="リダイレクト先の言語プレフィックス (例: en)")
    parser.add_argument("-o", "--output-yaml", default=DEFAULT_OUTPUT_YAML,
                        help="生成するリダイレクト設定YAMLファイル名")
    parser.add_argument("-d", "--redirect-dir", default=str(DEFAULT_REDIRECT_DIR),
                        help="リダイレクトHTMLファイルの出力先ディレクトリ")
    parser.add_argument("--skip-yaml-generation", action="store_true",
                        help="設定YAMLファイルの生成をスキップし、既存のファイルを使用します。")
    parser.add_argument("--skip-html-generation", action="store_true",
                        help="リダイレクトHTMLファイルの生成をスキップします。")

    args = parser.parse_args()

    # 1. ルール生成 (スキップしない場合)
    if not args.skip_yaml_generation:
        generated_rules = generate_redirect_rules(args.source_dir, args.lang_prefix)
        if generated_rules: # ルールが生成された場合のみ書き込み
             write_yaml_config(generated_rules, args.output_yaml)
        else:
             print("生成されたルールがないため、YAMLファイルは作成されませんでした。")
    else:
        print(f"設定ファイル '{args.output_yaml}' の生成をスキップします。")

    # 2. HTML生成 (スキップしない場合)
    if not args.skip_html_generation:
        generate_redirect_files(args.output_yaml, args.redirect_dir)
    else:
        print(f"リダイレクトHTMLファイル ({args.redirect_dir}) の生成をスキップします。")