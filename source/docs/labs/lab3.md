# Лабораторная работа 3: *CI/CD для статического сайта в SourceCraft*

## Цели

1. Освоить настройку автоматического развертывания статического сайта, построенного на `MkDocs`.
2. Научиться публиковать один и тот же сайт на двух платформах: **SourceCraft** и **GitHub Pages**.
3. Познакомиться с базовой настройкой CI/CD для автоматической сборки и публикации сайта.

## Задачи

В рамках лабораторной работы требовалось:

- Реализовать сценарий автоматического развертывания статического сайта на `MkDocs` с использованием **SourceCraft**
- Реализовать сценарий автоматического развертывания этого же сайта с помощью **GitHub Actions**
- Использовать один локальный репозиторий, в котором настроены два удаленных репозитория:
  - `origin` — репозиторий на GitHub
  - `sourcecraft` — репозиторий на SourceCraft

## Ход работы

### 1. Подготовка исходного проекта

В качестве основы использовался ранее созданный статический сайт на `MkDocs`, который уже содержал структуру страниц, тему оформления и материалы по лабораторным работам.

Структура проекта после выполнения лабораторной работы приобрела следующий вид:

```bash
breakthrough.github.io/
├── .github/
│   └── workflows/
│       └── mkdocs.yml
├── .sourcecraft/
│   ├── ci.yaml
│   └── sites.yaml
├── docs/
├── source/
│   └── mkdocs.yml
├── README.md
└── ...
```

### 2. Подключение SourceCraft

Для настройки публикации через **SourceCraft** были выполнены следующие действия:
1.	Выполнена авторизация на сайте sourcecraft.dev с использованием аккаунта Яндекс.
2.	Создана публичная организация.
3.	Внутри организации создан пустой публичный репозиторий `site-1`.
4.	Создан персональный токен доступа (PAT) для работы по HTTPS.
5.	Для токена были выданы права Maintaine (Ответственный за репозиторий).
6.	В уже существующий локальный репозиторий был добавлен второй удаленный репозиторий `sourcecraft`.

Для добавления удаленного репозитория использовалась команда:

```bash
git remote add sourcecraft https://<имя_аккаунта>:<персональный_токен>@git.sourcecraft.dev/<имя_аккаунта>/<имя_репозитория>.git
```

После этого была выполнена проверка списка удаленных репозиториев командой:

```bash
git remote -v
```

Вывод:

```bash
origin  https://github.com/breakthr0ugh/breakthrough.github.io.git (fetch)
origin  https://github.com/breakthr0ugh/breakthrough.github.io.git (push)
sourcecraft     https://токен.sourcecraft.dev/kiuyqu/site-1.git (fetch)
sourcecraft     https://токен.sourcecraft.dev/kiuyqu/site-1.git (push)
```

### 3. Публикация проекта в SourceCraft

После добавления второго удаленного репозитория локальный проект был отправлен в **SourceCraft**:

```bash
git push sourcecraft main
```

Далее были созданы специальные конфигурационные файлы платформы **SourceCraft** для автоматического развертывания:

1. Файл `.sourcecraft/sites.yaml`

Данный файл задает, из какой ветки и из какой директории **SourceCraft** должен публиковать сайт.

```bash
site:
  root: "/site"
  ref: "release"
```

??? info "Пояснение"
    SourceCraft Sites берет опубликованную версию сайта из ветки release -> папки /site.

2. Файл `.sourcecraft/ci.yaml`

Данный файл описывает сценарий автоматической сборки и публикации сайта.

Основная логика:
- при каждом `push` в ветку `main` запускается `workflow build-and-deploy-site`
- внутри `workflow`: 
    - устанавливаются `Python`, `mkdocs` и `mkdocs-material`
    - выполняется сборка сайта
    - создается или обновляется ветка `release`
    - готовые файлы сайта копируются в папку `site`
    - изменения отправляются в `release`

Код:

```bash
on:
  push:
    - workflows: [build-and-deploy-site]
      filter:
        branches: ["main"]

workflows:
  build-and-deploy-site:
    tasks:
      - name: build-and-publish
        cubes:
          - name: build-and-publish-site
            image: docker.io/library/python:3.12
            script:
              - |
                set -e

                apt-get update
                apt-get install -y git rsync

                python -m pip install --upgrade pip
                pip install mkdocs mkdocs-material

                git config --global user.name "SourceCraft CI"
                git config --global user.email "ci@sourcecraft.local"

                mkdocs build -f source/mkdocs.yml -d site

                rm -rf .release-worktree
                git fetch origin || true

                if git show-ref --verify --quiet refs/remotes/origin/release; then
                  git worktree add .release-worktree origin/release
                else
                  git worktree add --detach .release-worktree
                  cd .release-worktree
                  git checkout --orphan release
                  cd ..
                fi

                cd .release-worktree
                find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
                mkdir -p site
                rsync -a ../source/site/ ./site/

                git add .
                git commit -m "Deploy site from main" || echo "No changes to commit"
                git push origin HEAD:release

                cd ..
                git worktree remove .release-worktree --force
```

### 4. Настройка деплоя через GitHub Actions

Следующей частью лабораторной работы была настройка автоматической публикации того же сайта через **GitHub Actions** и **GitHub Pages**.
Для этого в репозитории была создана директория `.github//workflows` и файл `mkdocs.yml` внутри нее. В файле был описан процесс автоматического деплоя при каждом push в ветку main.

Полная конфигурация `workflow`:

```bash
name: Deploy MkDocs to GitHub Pages

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install mkdocs mkdocs-material

      - name: Build site
        run: mkdocs build -f source/mkdocs.yml

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: source/site

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

Логика `workflow`:
1. Проект клонируется в `runner`
2. Устанавливается Python 3.12
3. Устанавливаются `mkdocs` и `mkdocs-material`
4. Выполняется сборка сайта
5. Собранный сайт из папки `source/site` загружается как `artifact`
6. `Artifact` автоматически публикуется через **GitHub Pages**

## Выводы

В ходе выполнения лабораторной работы был реализован сценарий автоматического развертывания статического сайта на движке `MkDocs` сразу в двух средах: **SourceCraft** и **GitHub Pages**.

В рамках работы были освоены следующие практические действия:

- Настройка нескольких удаленных репозиториев в одном локальном проекте
- Создание и использование PAT для HTTPS-доступа
- Настройка публикации сайта через **SourceCraft** CI/CD
- Настройка публикации сайта через **GitHub Actions**
- Организация CI/CD-процесса для автоматической сборки и развертывания сайта

Ссылки на результаты:

- Статический сайт на **SourceCraft**:
*https://kiuyqu.sourcecraft.site/site-1/￼*
- Репозиторий в **SourceCraft**:
*https://sourcecraft.dev/kiuyqu/site-1?rev=main￼*
- Статический сайт на **GitHub Pages**:
*https://breakthr0ugh.github.io/￼*
- **GitHub**-репозиторий:
*https://github.com/breakthr0ugh/breakthrough.github.io￼*