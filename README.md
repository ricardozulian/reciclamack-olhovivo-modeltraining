# ReciclaMack Olho Vivo — Treinamento do Modelo

Pipeline de preparação de dados, validação e treinamento do modelo de visão computacional do projeto de extensão universitária **Olho Vivo — Identificação de Resíduos Eletroeletrônicos por Visão Computacional**, desenvolvido no âmbito da Universidade Presbiteriana Mackenzie, Faculdade de Computação e Informática (FCI).

O objetivo deste repositório é treinar e versionar um modelo YOLO11n para detecção de resíduos eletroeletrônicos em imagens reais, com exportação para ONNX para uso no backend.

## Contexto acadêmico

- Instituição: Universidade Presbiteriana Mackenzie
- Unidade: Faculdade de Computação e Informática (FCI)
- Área temática: Meio Ambiente, Tecnologia e Produção, Educação Ambiental
- Linha de extensão: Gestão de Resíduos Sólidos e Educação para a Sustentabilidade
- Coordenação/orientação: Profa. Sandra Bozolan

## Equipe discente

- Ricardo Zulian de Souza Amaral
- Marcos Volponi Cervan
- Flavio Estevam Nogueira Andrade

## Objetivo técnico

- Curar datasets públicos de resíduos eletroeletrônicos.
- Validar estrutura YOLO de imagens e bounding boxes.
- Treinar YOLO11n localmente no MacBook ou em ambiente com GPU.
- Exportar o melhor checkpoint para ONNX.
- Produzir métricas e manifestos de rastreabilidade.
- Entregar o artefato final ao backend para inferência em CPU.

## Dataset

Os dados não ficam versionados neste repositório. O fluxo previsto é baixar o dataset final do Kaggle para um diretório local ignorado pelo Git:

```powershell
kaggle datasets download <kaggle-user>/reciclamack-olhovivo-detection-v1 -p data/kaggle --unzip
```

Após o download, a estrutura esperada é:

```text
data/kaggle/reciclamack-detection-v1/data.yaml
data/kaggle/reciclamack-detection-v1/train/
data/kaggle/reciclamack-detection-v1/valid/
data/kaggle/reciclamack-detection-v1/test/
```

O arquivo `config/detection_local_train.kaggle.yaml` já aponta para esse layout.

## Treinamento local

Instale as dependências:

```powershell
pip install -r requirements.txt
```

Execute um teste curto de treinamento:

```powershell
python scripts/train_detection_local.py --config config/detection_local_train.kaggle.yaml
```

Para rodadas maiores, use os arquivos em `config/detection_local_train.*.yaml` como base e ajuste épocas, tamanho de imagem, batch e nome da rodada.

## Validação do dataset

```powershell
python scripts/validate_detection_dataset.py --dataset-root data/kaggle/reciclamack-detection-v1 --data-yaml data/kaggle/reciclamack-detection-v1/data.yaml
```

## Artefatos

Pesos `.pt`, exportações `.onnx`, datasets, caches e rodadas de treinamento não devem ser commitados. Apenas scripts, configurações, testes, documentação e manifestos leves entram no Git.

## Papel no sistema

Este repositório é autônomo e cobre o ciclo de treinamento e validação do modelo. A API de inferência e a aplicação web ficam em repositórios separados.
