// language_switch.js

document.addEventListener('DOMContentLoaded', function () {
  // --- 設定項目 (必要に応じて調整) ---
  const languages = ['en', 'ja']; // 対応言語のリスト
  const defaultLang = 'en';     // デフォルトまたはベースとなる言語
  // サイトのベースパス (例: https://<user>.github.io/<repo>/ の /<repo>)
  // GitHub Pagesでリポジトリ名がパスに含まれる場合は設定。カスタムドメインなら ''
  const basePath = '';
  // リンクを挿入する場所のCSSセレクタ (テーマによって調整が必要)
  // sphinx-book-theme の場合、サイドバーのロゴやタイトル部分の後ろあたり
  const targetSelector = 'a.navbar-brand.logo';
  // ------------------------------------

  const currentPath = window.location.pathname; // 例: /ommx/ja/user_guide/page.html
  let currentLang = defaultLang;
  let otherLang = languages.find(lang => lang !== defaultLang); // 切り替え先の言語候補

  // 現在のパスから言語を判定
  for (const lang of languages) {
    if (currentPath.startsWith(`${basePath}/${lang}/`)) {
      currentLang = lang;
      otherLang = languages.find(l => l !== lang);
      break; // 言語が見つかったらループを抜ける
    }
  }

  if (!otherLang) {
    console.warn('Could not determine the other language.');
    return; // 切り替え先言語が不明な場合は何もしない
  }

  // 切り替え先のURLを生成
  let otherLangUrl;
  const currentLangPrefix = `${basePath}/${currentLang}/`;
  const otherLangPrefix = `${basePath}/${otherLang}/`;

  if (currentPath.startsWith(currentLangPrefix)) {
    // 現在のパスが言語プレフィックスで始まっている場合、それを置換
    otherLangUrl = currentPath.replace(currentLangPrefix, otherLangPrefix);
    // クエリパラメータやハッシュがあれば維持
    otherLangUrl += window.location.search + window.location.hash;
  } else {
    // 現在のパスが言語プレフィックスで始まっていない場合 (例: /ommx/protobuf.html)
    // 切り替え先言語のトップページに遷移させる (仕様に応じて変更可能)
    console.log(`Current path '${currentPath}' does not start with expected language prefix '${currentLangPrefix}'. Falling back to root of other language.`);
    otherLangUrl = otherLangPrefix;
  }


  // リンクを表示する要素を取得
  const targetElement = document.querySelector(targetSelector);

  if (targetElement) {
    // リンク要素を作成
    const langLink = document.createElement('a');
    langLink.href = otherLangUrl;
    langLink.textContent = (otherLang === 'ja') ? '日本語' : 'English'; // 表示テキスト
    langLink.className = 'language-switch-button'; // CSSでスタイルを当てるためのクラス名

    // スタイルを直接設定 (任意)
    langLink.style.display = 'block';
    langLink.style.textAlign = 'center';
    langLink.style.marginTop = '1em'; // 上に少し余白
    langLink.style.padding = '0.5em';
    langLink.style.border = '1px solid #ccc'; // 枠線
    langLink.style.borderRadius = '4px'; // 角丸


    // ★重要：見つけたロゴ要素の「親要素」の後ろにリンクを挿入する
    targetElement.parentElement.insertAdjacentElement('afterend', langLink);

    // // ターゲット要素の後ろにリンクを挿入
    // targetElement.insertAdjacentElement('afterend', langLink);
    // console.log(`Language switch link added: -> ${otherLangUrl}`);

  } else {
    console.warn(`Language switcher target element ('${targetSelector}') not found.`);
  }
});