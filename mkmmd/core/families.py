"""Secondary-motion chain families from bone names (Japanese, Chinese, English and Spanish conventions of MMD models).

Families group physics-driven chains so tools can treat them alike (one set of hair settings for every back-hair
chain). Names are NFKC-normalised first (half-width katakana such as ｽｶｰﾄ become スカート). The first rule that
matches wins, so specific kinds come before generic ones. Rules were checked against a corpus of 178 models."""
import re
import unicodedata

_W = r"(?<![a-z])"          # no ASCII letter right before (bone names join words with _ . or CamelCase)
_E = r"(?![a-z])"
RULES = [
    ("bangs", [r"前髪", r"前髮", r"bang", r"fringe", r"ahoge", r"agohe", r"アホ毛", r"あほげ", r"アホゲ", r"flequillo",
               _W + r"fleq"]),
    ("side_hair", [r"横髪", r"横髮", r"もみあげ", r"モミアゲ", r"揉み上げ", _W + r"momi", r"side.?hair", r"sideburn",
                   r"patilla", r"^サイド\d*$|^サイド[\d_.]"]),
    ("back_hair", [r"後髪", r"後髮", r"後ろ髪", r"後毛", r"back.?hair", r"nuca", r"后长发"]),
    ("twintail", [r"ツイン", r"twin", r"テール(?!.*尻)"]),
    ("braid", [r"三つ編", r"みつあみ", r"三編", r"おさげ", r"お下げ", r"braid", r"pigtail", r"辫", _W + r"osage" + _E,
               _W + r"osg" + _E]),
    ("hair", [r"髪", r"髮", r"hair", r"毛", r"pony", r"ポニテ", r"coleta", _W + r"pelo", r"pelito", _W + r"pel" + _E,
              r"ドリル", _W + r"drill", _W + r"flick"]),
    ("ears", [r"耳", _W + r"ears?" + _E, _W + r"mimi" + _E]),
    ("tail", [r"尻尾", r"しっぽ", r"シッポ", r"尾", r"tail"]),
    ("skirt", [r"スカート", r"skirt", r"裾", r"前垂", r"後垂", r"袴", _W + r"skt?" + _E, r"ヒラヒラ", r"スリット"]),
    ("ribbon", [r"リボン", r"りぼん", r"ribbon", _W + r"rib" + _E, r"紐", r"ひも", r"ネクタイ", r"タイ(?!ツ)",
                _W + r"tie" + _E, r"(?<!el)bow" + _E, r"蝴蝶结", r"コード", r"ベルト", r"belt", r"strap", r"correa",
                r"帯", r"系绳", r"マフラー", r"スカーフ", r"scarf", r"shawl", r"bandage", _W + r"lace" + _E]),
    ("breasts", [r"胸", r"乳", r"おっぱい", r"bust", r"breast"]),
    ("sleeve", [r"袖", r"sleeve", r"cuff"]),
    ("coat", [r"コート", r"マント", r"ケープ", r"上着", r"coat", r"cape", r"mantle", r"cloak", r"chaqueta", r"jacket",
              r"ジャケット", r"poncho", r"パーカー", r"parka", r"cardigan", r"シャツ", r"shirt", r"上服", r"uwagi",
              r"襟", r"エリ", r"collar", r"cuello", r"セーラ", r"フード", _W + r"hood"]),
    ("accessory", [r"飾", r"アクセ", _W + r"acc", r"帽", _W + r"hat" + _E, r"花", r"羽", r"wing", r"鈴", r"bell",
                   r"chain", r"チェーン", r"チェイン", r"鎖", r"cadena", r"イヤリング", r"earring", r"首輪", r"ランドセル",
                   r"タグ", r"ジップ", r"ファスナ", _W + r"zip", _W + r"flag", r"坠饰"]),
]
_COMPILED = [(fam, [re.compile(p, re.IGNORECASE) for p in pats]) for fam, pats in RULES]
FAMILIES = [f for f, _ in RULES] + ["other"]
HAIR_FAMILIES = {"bangs", "side_hair", "back_hair", "twintail", "braid", "hair"}


def classify(*names):
    """Family of a chain from its bone names (root first). Falls back to `other`."""
    for name in names:
        if not name:
            continue
        n = unicodedata.normalize("NFKC", name)
        for fam, pats in _COMPILED:
            if any(p.search(n) for p in pats):
                return fam
    return "other"
