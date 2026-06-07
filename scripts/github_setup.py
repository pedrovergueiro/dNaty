"""
Configura repo GitHub: topics, description, homepage.
Uso: python scripts/github_setup.py <token>
Gera visibilidade no GitHub Explore.
"""
import sys
import json
import urllib.request
import urllib.error

OWNER = "pedrovergueiro"
REPO  = "dNaty"

TOPICS = [
    "pytorch",
    "machine-learning",
    "neural-architecture-search",
    "model-compression",
    "edge-ml",
    "nas",
    "onnx",
    "evolutionary-algorithm",
    "python",
    "deep-learning",
    "iot",
    "embedded-ml",
    "cpu-inference",
    "nsga2",
]

DESCRIPTION = "Compress PyTorch models for edge devices — CPU-only, no GPU, no retraining. One function call."
HOMEPAGE    = "https://dnaty.org"


def api(token: str, method: str, path: str, data: dict | None = None):
    url = f"https://api.github.com{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/github_setup.py <github_token>")
        print("Token: github.com/settings/tokens (precisa de repo scope)")
        sys.exit(1)

    token = sys.argv[1]
    print(f"Configurando {OWNER}/{REPO} ...")

    # 1. Topics
    print(f"\n[1/3] Adicionando {len(TOPICS)} topics ...")
    r = api(token, "PUT", f"/repos/{OWNER}/{REPO}/topics", {"names": TOPICS})
    if r:
        print(f"  OK: {r.get('names', [])}")

    # 2. Description + homepage
    print("\n[2/3] Atualizando description e homepage ...")
    r = api(token, "PATCH", f"/repos/{OWNER}/{REPO}", {
        "description": DESCRIPTION,
        "homepage": HOMEPAGE,
        "has_wiki": False,
        "has_issues": True,
        "has_discussions": True,
    })
    if r:
        print(f"  OK: description='{r.get('description')}'")
        print(f"  OK: homepage='{r.get('homepage')}'")

    # 3. Verifica
    print("\n[3/3] Verificando ...")
    r = api(token, "GET", f"/repos/{OWNER}/{REPO}")
    if r:
        print(f"  stars: {r.get('stargazers_count')}")
        print(f"  forks: {r.get('forks_count')}")
        print(f"  topics: {r.get('topics')}")
        print(f"  description: {r.get('description')}")

    print("\nPronto! O repo agora aparece nas buscas e categorias do GitHub Explore.")
    print("Para criar o Release v1.1.0:")
    print("  github.com/pedrovergueiro/dNaty/releases/new")


if __name__ == "__main__":
    main()
