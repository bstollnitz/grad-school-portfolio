# Predicting time evolution of a pigment in water

## Prerequisites

1. Download and install [Python 3](https://www.python.org/downloads/) if you don't have it already.

1. Change to the `PigmentInWater` directory.

1. Download all the [data files and videos](https://1drv.ms/f/s!AiCY1Uw6PbEfhaEncETEzJ0kakis6g) and put them in the `Data` subdirectory.

1. Install [Chromedriver](http://chromedriver.chromium.org/getting-started) by 
following the setup steps, including adding the executable to your path.

1.	Install [orca](https://github.com/plotly/orca) by following one of the three installation methods on their web page.

1. Create a Python virtual environment, activate it, and install the packages we need:

    (MacOS)
    ```sh
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    (Windows)
    ```sh
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Running

```sh
python main.py
```
