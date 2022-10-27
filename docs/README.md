## Build Documentation

1. Clone MMEval

   ```bash
   git clone https://github.com/open-mmlab/mmeval.git
   cd mmeval
   ```

2. Install the building dependencies of documentation

   ```bash
   pip install -r requirements/docs.txt
   ```

3. Change directory to `docs/en` or `docs/zh_cn`

   ```bash
   cd docs/en  # or docs/zh_cn
   ```

4. Build documentation

   ```bash
   make html
   ```

5. Open `_build/html/index.html` with browser
