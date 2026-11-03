# Síntese comparativa — PCB, NEU e NASA FD001

[← Visão geral](../README.md) · [PCB](01-pcb.md) · [NEU](02-neu.md) · [NASA](03-nasa-fd001.md)

## Comparação transversal: classificação PCB versus NEU

o que mudou do PCB para o NEU?

| Aspecto | PCB | NEU |
|---|---|---|
| Unidade de análise | Recorte derivado de anotação XML | Imagem integral de textura |
| Melhor teste nesta execução | CNN · GridSearchCV | MLP · Optuna |
| Acurácia final | **93,73%** | **88,33%** |
| Principal dificuldade | Pequenos padrões geométricos em regiões recortadas | Texturas sutilmente semelhantes |
| Cuidado de validação | Separar recortes por imagem de origem | Testar em imagens não utilizadas na busca |

A diferença de desempenho **não é evidência de que o PCB seja intrinsecamente mais fácil**: a unidade de análise e o pipeline variam. No PCB, o modelo recebe recortes já localizados, enquanto o enunciado exigia trabalhar sobre a placa completa. No NEU, a principal fragilidade observada é a diferença entre a CNN de maior acurácia CV e seu desempenho final em teste. Em ambos os casos, mais do que eleger um “campeão”, o experimento mostra por que F1 por classe, matriz de confusão e particionamento independente são indispensáveis.

## O que faria em uma próxima etapa?

Seria importante (i) executar a detecção/localização de defeitos diretamente na **placa PCB inteira**; (ii) comparar 64 × 64 contra as resoluções recomendadas no guia; (iii) aproximar o número de avaliações e espaços de busca de GS e Optuna; (iv) repetir treinamentos com várias sementes e intervalos de confiança; e (v) revisar os erros das classes mais frágeis antes de considerar uma aplicação em produção.

## Por que a regressão NASA não pode ser comparada por acurácia?

A tarefa NASA prevê uma **variável contínua (RUL em ciclos)**, e não seis classes. Por isso, suas métricas são RMSE, MAE e NASA Score. A LSTM escolhida alcançou **RMSE de 14,52 ciclos** em 100 motores oficiais de teste. O erro foi menor na faixa de RUL até 30 ciclos (MAE 3,19) e maior na faixa acima de 130 (MAE 20,44, n=8), o que exige cautela ao discutir previsões para motores ainda distantes da falha.

## Conclusão geral

Os três experimentos reforçam que **representação dos dados, separação correta entre treino/validação/teste, orçamento da busca e análise de erro por classe ou faixa** são essenciais para avaliação rigorosa. Os resultados são de execuções registradas nos notebooks, não de treinamentos refeitos durante a organização deste repositório.
