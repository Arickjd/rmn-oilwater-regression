# Trilha 1: Nuclear Magnetic Resonance (NMR)
**Previsão da proporção óleo–água a partir de curvas de relaxação magnética usando modelos de IA**

---

## Visão geral

Este projeto tem como objetivo desenvolver **modelos de IA** capazes de prever as **proporções de fluidos (óleo e água)** em rochas a partir de curvas de relaxação magnética obtidas por sequências **CPMG**.  A variável alvo é a **fração de óleo (%)** correspondente a cada curva.

---

## Descrição dos dados

### Formato do arquivo carregado
O dataset está em **formato tabular**. Cada linha representa uma curva M⊥(t) e a última coluna é a saturação, (para o dado de teste, não possui a coluna de saturação).
Na etapa de pré-processamento, devido a natureza do sinal de decaimento, selecionaremos apenas o sinal até o tempo $t_{700}$

Exemplo de esquema:

| t₁ | t₂ | t₃ | ... | $t_{700}$ | oil_fraction |
|----|----|----|-----|------|--------------|

- **t₁..$t_{700}$**: amostras temporais (valores de M⊥(t) com ruído)
- **oil_fraction**: alvo em porcentagem


### Proporções de fluido
- Fração de óleo varia entre **0% e 80%**

---

## Bibliotecas sugeridas e ambiente

- requirements.txt
- Ambiente PyTorch (CUDA 12.4)
