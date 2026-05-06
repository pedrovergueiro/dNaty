"""
Runner principal — executa todos os experimentos e gera o relatório.
Uso: python run_experiments.py
"""
import sys
import os

# Garantir que o pacote dnaty está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("dNaty — Experimentos Completos")
print("=" * 60)

print("\n[1/3] Experimento 1 — MNIST e FashionMNIST")
from dnaty.experiments.exp1_mnist import main as run_exp1
run_exp1()

print("\n[2/3] Experimento 3 — Split-MNIST Continual Learning")
from dnaty.experiments.exp3_cl import main as run_exp3
run_exp3()

print("\n[3/3] Gerando relatório Markdown...")
from dnaty.analysis.report import generate_report
generate_report()

print("\n" + "=" * 60)
print("Concluído! Arquivos gerados:")
print("  results/exp1_results.json")
print("  results/exp3_cl_results.json")
print("  DNATY_RESULTS.md")
print("=" * 60)
