uv run jupyter book build ja
uv run jupyter book build en
# _build/htmlがあれば中身を削除、なければ作成
if [ -d "_build/html" ]; then
    rm -rf _build/html/*
else
    mkdir -p _build/html
fi

cp -rf redirect/* _build/html

uv run python generate_redirects.py en en -d _build/html

mkdir -p _build/html/ja
mkdir -p _build/html/en

mv ja/_build/html/* _build/html/ja
mv en/_build/html/* _build/html/en
