# Regressão de saturação de água com RMN

Pipeline Python para prever `Swirr_PHIX` a partir de curvas de RMN, extraído de
`preprocessing.ipynb` e `modeling.ipynb` da raiz. Os notebooks foram preservados
como referência dos experimentos.

## Organização

```text
main.py                       # Pipeline completo e argumentos de execução
requirements.txt              # Dependências do pipeline Python
preprocess/
    data.py                   # Leitura/validação do CSV e seleção dos ecos
    groups.py                 # Mapeamento explícito de amostra para grupo
    signal.py                 # DWT, min-max, truncamento e média por bin
    pipeline.py               # Encadeamento do pré-processamento
    split.py                  # Treino/validação/teste e PCA
models/
    data_augmentation.py      # MixUp físico somente no treino
    mlp.py                    # MLP com PCA (célula 14)
    pinn.py                   # PINN original (célula 17)
    pinn_log_t2.py             # PINN com log(T2) (célula 18)
    pinn_weighted.py           # PINN com Sw direto e perda ponderada (célula 19)
    common.py                 # Arquitetura/perda comuns às PINNs
    training.py               # Treinamento, predição e métricas
    registry.py               # Modelos/entradas compartilhados entre CV e teste
    cross_validation.py       # CV repetida com validação interna e grupos
    evaluation.py             # Bootstrap, comparação pareada e relatório
func_plots/
    evaluation.py             # Scatter 1x4 e curvas de treinamento
    preprocessing.py          # Distribuições, sinais, DWT e PCA
    utils.py                  # Exportação PNG (300 dpi) e PDF vetorial
    validation.py             # Dispersão dos folds e IC das métricas de teste
tests/
    test_pipeline.py          # Verificações numéricas e isolamento dos dados
    test_validation.py        # Isolamento de grupos, cobertura OOF e bootstrap
```

Cada pacote contém `__init__.py`. Importar módulos não inicia treinamento nem
abre janelas. Os notebooks em `notebooks/` e demais experimentos permanecem
como estavam.

## Instalação e execução

Use Python 3.10 ou superior. No PowerShell, a partir da raiz:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python main.py --groups-file sample_groups.csv
```

O dataset contém curvas relacionadas por grupo. Antes de executar, identifique
esses grupos em `sample_groups.csv`: colunas `sample_id` (linha do CSV original,
base zero) e `group` (amostra física, poço ou simulação original). Todas as curvas
relacionadas devem compartilhar o identificador de grupo. O arquivo
`sample_groups.template.csv` contém os 575 IDs e grupos em branco para preencher;
é um modelo de preenchimento, **não um mapeamento válido**. Não deduza os grupos
pelo alvo, nem atribua um grupo diferente por curva para contornar a separação.

Exemplo de formato, ilustrativo:

```csv
sample_id,group
0,amostra_A
1,amostra_A
2,amostra_B
```

O mapeamento real deve cobrir exatamente todas as linhas. A CLI exige
`--groups-file` ou uma declaração explícita `--independent-samples`; esta última
é destinada a **outros datasets com independência conhecida**, não ao dataset
agrupado deste projeto. Sem a origem dos grupos, não é possível calcular uma
nova avaliação confiável para estes dados.

Por padrão, lê `RMN_data/GulfCoast_RMN_Synthetic.csv`, executa CV 5×3, treina os
quatro modelos finais e salva em `output/pipeline/`. Caminhos padrão são relativos ao `main.py`;
caminhos fornecidos na CLI são relativos ao diretório de execução. Executar
novamente no mesmo diretório de saída substitui arquivos gerados de mesmo nome;
use `--output-dir` para separar experimentos.

Execução curta de integração (não representa convergência):

```powershell
.venv/Scripts/python main.py --groups-file sample_groups.csv --cv-folds 2 --cv-repeats 1 --mlp-epochs 2 --pinn-epochs 2 --output-dir output/smoke_test
```

Para reutilizar um CSV já processado, pulando DWT/normalização/compressão:

```powershell
.venv/Scripts/python main.py --input model_input.csv --preprocessed --groups-file sample_groups.csv
```

Consulte `main.py --help` para épocas, seed, tamanhos dos conjuntos, PCA,
parâmetros de sinal e dispositivo. `--augmented-size 0` desativa MixUp;
`--no-plots` desativa figuras; `--device cpu` força CPU. `--cv-folds 0` desativa
somente CV, mantendo separação por grupo e bootstrap do teste.

## Etapas e modelos

1. **Sinais:** DWT `db6`, nível 6, threshold soft com estimativa de ruído do
   notebook; min-max individual; remoção dos últimos 1.000 ecos; média de cada
   4 ecos. Os dados fornecidos passam de 3.000 para 500 ecos. Sinais constantes
   são normalizados para zero. As quatro propriedades físicas são preservadas.
2. **Divisão:** reserva 30% dos grupos para teste (`--test-group-fraction 0.3`).
   Do desenvolvimento, reserva 20% dos grupos para validação interna. Os números
   de curvas dependem dos tamanhos dos grupos. `--test-size 175` aplica-se apenas
   ao modo de linhas independentes; nesse modo legado, seriam 320/80/175 curvas.
3. **MixUp:** amplia o treino para 1.000 amostras. Interpola ecos, `MBVI`, `MPHI`
   e `PHIX` de duas amostras reais distintas e recalcula `Swirr_PHIX = MBVI / PHIX`
   para cada amostra sintética. Não reduz o treino se já superar o tamanho pedido.
4. **PCA:** ajustado apenas no treino aumentado, retendo 99,9% da variância.
   O MLP recebe componentes; as PINNs recebem os 500 ecos processados.
5. **Treino e avaliação:** Adam, learning rate 0,001, batch 256; 40 épocas no
   MLP e 120 em cada PINN. Restaura a melhor época pela perda total de validação.
   Calcula RMSE, MAE e R² na validação e no teste.

## Validação cruzada e confiança nas métricas

- **CV externa:** 5 folds × 3 repetições somente nos grupos de desenvolvimento.
  Grupos são embaralhados com seeds fixas e distribuídos em folds com quantidades
  semelhantes de grupos; o número de curvas pode variar. É preciso ter pelo
  menos 5 grupos no desenvolvimento para a configuração padrão.
- **Validação interna:** cada treino externo é novamente separado por grupos.
  A validação interna escolhe a época e controla o scheduler. O fold externo
  não participa da seleção. MixUp e PCA são refeitos usando só o treino interno.
  Trata-se de CV externa com holdout interno, sem busca de hiperparâmetros por
  nested K-fold. Arquiteturas e demais hiperparâmetros ficam fixos.
- **Predições OOF:** cada curva é avaliada uma vez por repetição por um modelo
  que nunca viu seu grupo no treino nem na validação interna. Métricas são
  calculadas separadamente por repetição, sem tirar média de predições para
  simular um ensemble. Média e desvio-padrão dos folds/repetições são descritivos;
  não usamos desvio/raiz(n), pois os ajustes compartilham dados.
- **Seleção:** menor RMSE OOF médio por repetição, incluindo a referência que
  prevê a média dos alvos reais de treino. A escolha é registrada antes da
  avaliação final. O desempenho CV do vencedor pode ser otimista pela seleção
  entre candidatos; o teste é a avaliação separada, respeitado o histórico dos dados.
- **Teste:** bootstrap pareado por grupo, 2.000 reamostragens e IC percentil de
  95% para RMSE, MAE e R². Um grupo sorteado leva todas as suas curvas. É preciso
  reservar ao menos dois grupos; poucos grupos limitam a precisão dos intervalos.
  `--bootstrap-resamples` e `--confidence` configuram esses parâmetros.

Os IC do teste são condicionais aos modelos ajustados: quantificam a incerteza
de amostragem dos grupos de teste, não toda a variabilidade de treino/seleção.
Não são intervalos de predição de uma curva. As métricas ponderam cada curva
igualmente; grupos maiores contribuem com mais curvas. Reamostras de alvo
constante têm R² indefinido, excluído apenas do IC de R² e contabilizado.

As diferenças pareadas A − B entre modelos são exploratórias, sem correção por
múltiplas comparações. Para RMSE/MAE, uma diferença negativa favorece A; para
R², positiva favorece A. Não interprete apenas o melhor valor pontual como
evidência de superioridade.

**Limite do histórico:** as métricas antigas foram calculadas por linhas e
podem ser otimistas, pois há curvas do mesmo grupo. Os dados já foram consultados
nos notebooks e na execução anterior; uma nova separação não transforma esses
dados em uma validação externa inédita. A confirmação para o artigo requer
grupos novos e não consultados, com o protocolo fixado antes da avaliação.

Referências: [seleção e avaliação separadas](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
[CV e grupos](https://scikit-learn.org/stable/modules/cross_validation.html) e
[bootstrap pareado/percentil](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

| Modelo | Arquitetura | Física e perda total |
| --- | --- | --- |
| MLP + PCA | entrada → 256 → 128 → 1, ReLU | MSE do alvo; saída linear |
| PINN original | ecos → 128 → 32 → 4, ReLU | amplitudes sigmoid, T2 softplus, tempo em bins; MSE alvo + 4 × MSE físico |
| PINN log(T2) | ecos → 128 → 32 → 4, ReLU | amplitudes sigmoid, T2 exponencial com clamp, tempo normalizado; MSE alvo + 4 × MSE físico |
| PINN ponderada | ecos → 128 → 32 → 4, ReLU | Sw sigmoid, M0 softplus, T2 curto < longo; 20 × MSE alvo + erro físico ponderado por exp(-4t) |

As duas primeiras PINNs preservam a razão da primeira amplitude pela soma das
amplitudes usada no notebook. A última associa Sw ao componente de T2 curto.
O tempo representa índices de bins (original) ou índices divididos pelo número
de bins (outras variantes), sem calibração em unidades físicas de tempo.

**Ajustes metodológicos:** validação separada antes do MixUp e PCA evita que
amostras reservadas ou suas misturas participem do ajuste. O teste passa a ser
avaliado explicitamente pelos quatro modelos. Nas variantes com scheduler,
`ReduceLROnPlateau` usa os mesmos pesos da perda total de treino. O checkpoint de
menor perda total de validação é restaurado. Portanto, as métricas não devem
reproduzir exatamente as células originais. Seeds controlam os sorteios, mas
resultados podem variar entre dispositivos e versões das bibliotecas.

## Resultados gerados

- `model_input.csv`: ecos processados e propriedades físicas.
- `train_augmented.csv` e `splits.csv`: treino ampliado e identificação das
  amostras reais por conjunto (índice da linha original, base zero).
- `metrics.csv` e `predictions_test.csv`: métricas e predições dos quatro modelos,
  mais a referência da média, alinhadas por `sample_id` e com o grupo identificado.
- `cross_validation/`: partições auditáveis, métricas por fold/repetição,
  predições OOF, históricos de treino, resumos e `selection.json`.
- `test_confidence_intervals.csv` e `test_paired_comparisons.csv`: IC do teste
  e diferenças entre modelos, com contagem de grupos e reamostras válidas.
- `evaluation_report.md`: protocolo, resultados e limites de interpretação.
- `history_*.csv` e `config.json`: perdas, configuração e melhor época por modelo.
- `checkpoints/`: pesos PyTorch `.pt` com dimensões/colunas e PCA em `pca.joblib`.
- `figures/`: scatter 1×4, históricos, distribuições, PCA, sinais aumentados e
  comparação DWT, em PNG/PDF. A figura DWT é gerada ao processar o CSV bruto.
  Inclui também scatter OOF da primeira repetição, dispersão dos folds e IC do teste.

Para carregar um modelo, instancie a classe correspondente com `input_dim` do
checkpoint e aplique `load_state_dict(checkpoint['state_dict'])`. Para o MLP,
transforme os ecos processados com o PCA salvo antes da predição. A ordem dos
ecos está em `echo_columns`; propriedades físicas não são features.

## Figuras reutilizáveis

```python
import pandas as pd
from func_plots import plot_model_scatter_grid, save_figure

pred = pd.read_csv("output/pipeline/predictions_test.csv")
fig, axes = plot_model_scatter_grid(pred["y_true"], {
    "MLP + PCA": pred["mlp"],
    "PINN original": pred["pinn"],
    "PINN log(T2)": pred["pinn_log_t2"],
    "PINN ponderada": pred["pinn_weighted"],
})
save_figure(fig, "output/pipeline/figures/comparacao_artigo")
```

As funções retornam figura e eixos para ajustes editoriais. O scatter usa
limites comuns, reta ideal e métricas por modelo.

## Verificação

```powershell
.venv/Scripts/python -m unittest discover -s tests -v
```

Os testes verificam pré-processamento, relação física do MixUp, isolamento de
amostras e grupos, PCA ajustado no treino, perdas físicas, cobertura OOF,
bootstrap de grupos de tamanhos diferentes e tratamento do R² indefinido.
