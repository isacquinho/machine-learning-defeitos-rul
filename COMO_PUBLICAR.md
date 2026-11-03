# Como publicar este projeto no GitHub

[← README](README.md)

Este pacote já está organizado como **conteúdo de um repositório**. Não é preciso converter Markdown: o GitHub renderiza `README.md`, páginas em `docs/` e imagens referenciadas pelos caminhos relativos.

## Caminho fácil: site do GitHub

1. Entre em [github.com](https://github.com) e autentique-se.
2. Crie um novo repositório com nome sugerido **`machine-learning-defeitos-rul`**; escolha **Public** se quiser que o professor leia sem entrar na sua conta, ou **Private** se quiser controlar o acesso.
3. Não marque as opções para criar README, `.gitignore` ou licença automaticamente, pois esses arquivos já estão no pacote.
4. **Extraia o ZIP** recebido. No repositório novo, escolha **Add file → Upload files** e envie o **conteúdo da pasta extraída**, mantendo a estrutura de subpastas (`docs/`, `notebooks/`, `assets/figuras/` e `apresentacao/`). Verifique se a interface preservou a estrutura; se não, use GitHub Desktop ou Git via terminal.
5. Confirme o upload e abra `README.md` na página principal do repositório. Teste os links dos relatórios e das imagens.
6. Compartilhe com o professor a URL principal do repositório, **não** o endereço do ZIP enviado ao ChatGPT.

## Alternativa pelo Git (mais confiável para pastas)

Crie um repositório **vazio** na interface do GitHub e, dentro da pasta extraída, execute:

```bash
git init
git add .
git commit -m "Publica estudos PCB, NEU e NASA FD001"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/machine-learning-defeitos-rul.git
git push -u origin main
```

Substitua `SEU_USUARIO` por seu usuário real. O push requer autenticação do GitHub, por gerenciador de credenciais, token ou SSH conforme seu ambiente.

## Recomendações antes de tornar público

- Verifique os **termos de uso dos datasets** e não suba os dados originais sem autorização.
- Leia os notebooks para conferir **caminhos pessoais, tokens e outros segredos** antes de publicar. Uma verificação automatizada inicial deste pacote não encontrou e-mails nem padrões óbvios de tokens nas células de código; ela não substitui revisão manual.
- Os notebooks preservam outputs do treinamento, mas executá-los de novo pode levar tempo e produzir resultados ligeiramente diferentes.
- Acrescente uma licença apenas quando decidir os termos de reutilização **do seu código/texto**; isso não concede direitos sobre dados, figuras ou bibliografia de terceiros.
- Se quiser um site navegável tipo artigo, pode ativar **GitHub Pages** posteriormente, mas não é obrigatório para entregar o relatório.

## Sobre publicação automática por este chat

Criar a pasta e o ZIP aqui **não cria** um repositório remoto no GitHub. Para publicação diretamente na sua conta, é necessário conectar uma integração autorizada ou realizar o upload autenticado por você.
