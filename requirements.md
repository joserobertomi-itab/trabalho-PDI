# Trabalho Prático 1 – Segmentação de Embalagens de Produtos Avícolas

## Objetivo

Desenvolver uma solução capaz de localizar e segmentar automaticamente embalagens de produtos derivados de frango presentes em imagens capturadas em ambiente industrial.

Nesta primeira etapa do projeto, não será realizada classificação/reconhecimento de texto (OCR). O foco exclusivo é a **segmentação das embalagens** presentes nas imagens.

O conjunto de dados contém imagens de diferentes produtos, incluindo embalagens dos tipos **bandeja** e **selado**, distribuídas em diversas classes de produtos. Embora as pastas possuam o nome do produto, essa informação será utilizada apenas para organização dos resultados.

---

## Classes Presentes na Base

- Asas Resfriado Selado
- Meio das Asas Congelado
- Coxinhas das Asas Congelado
- Coxinhas das Asas Congelado Selado
- Meio das Asas Congelado Selado
- Coxinhas das Asas Resfriado Selado
- Filé de Peito Congelado
- Filezinho Sassami Resfriado Selado
- Filezinho Sassami Congelado
- Filé de Peito Congelado Selado
- Coração
- Moela
- Peito Congelado
- Peito Resfriado
- Filé de Coxas e Sobrecoxas com Pele Congelado Selado
- Filé de Coxas e Sobrecoxas com Pele Congelado
- Coxas e Sobrecoxas Congelado
- Coxas e Sobrecoxas Resfriado Selado

---

## Descrição do Problema

As imagens apresentam desafios típicos de ambientes industriais, incluindo:

- Reflexos
- Deformações da embalagem
- Variações de iluminação
- Diferenças de posicionamento e orientação dos produtos

O programa deverá percorrer todas as imagens existentes em todas as pastas do conjunto de dados, detectar a região correspondente à embalagem do produto com o nome, gerando uma ou mais imagens contendo apenas a parte da embalagem segmentada.

> **Atenção:** Falsos positivos serão considerados como erro, ou seja, imagens produzidas sem o nome do produto, ou com nomes irrelevantes, como "só frango". As imagens segmentadas podem possuir uma parte do nome do produto (por exemplo, "asas" em uma imagem e "resfriada" em outra). Também não é necessário que as palavras estejam visíveis, apenas que a segmentação esteja na posição esperada.

---

## Estrutura de Arquivos

### Entrada

```
dataset/
├── Peito_Congelado/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── Moela/
└── ...
```

### Saída

> **Nota:** neste repositório a pasta de saída foi padronizada como `result/` (inglês).

```
result/
├── Peito_Congelado/
│   ├── img001_segmentada_1.png
│   ├── img001_segmentada_2.png
│   ├── img002_segmentada_1.png
│   ├── img003_segmentada_1.png
│   └── ...
├── Moela/
└── ...
```

---

## Restrições e Técnicas Permitidas

**Não é permitido** utilizar conteúdos ainda não abordados na disciplina, nem bibliotecas que realizem segmentação automaticamente por IA.

**É permitido** qualquer técnica baseada exclusivamente no conteúdo estudado na **Parte 1** da disciplina, incluindo:

- Espaços de cores
- Limiarização
- Segmentação
- Operações morfológicas
- Filtragem espacial
- Transformações geométricas
- Histogramas

Algoritmos adicionais poderão ser utilizados desde que pertençam aos tópicos abordados na Parte 1 da disciplina.

---

## Critério de Avaliação

| Critério | Peso |
|---|---|
| Desempenho na Base Fornecida | 60% |
| Avaliação Complementar (imagens inéditas) | 40% |

### Detalhes

1. **Desempenho na Base Fornecida (60%):** O algoritmo será executado sobre o conjunto de imagens disponibilizado.

2. **Avaliação Complementar (40%):** No dia da apresentação, o algoritmo será executado em um conjunto de imagens inéditas, não disponibilizadas previamente. Nenhuma modificação poderá ser realizada no código durante a avaliação. O objetivo é medir a capacidade de **generalização** da solução.

---

## Entregáveis

- Submeter o **link do Colab** no Moodle, **ou** um **Docker Compose** com a solução.
- Compartilhar o link do Colab com: [alessandro.rodrigues@ifg.edu.br](mailto:alessandro.rodrigues@ifg.edu.br)