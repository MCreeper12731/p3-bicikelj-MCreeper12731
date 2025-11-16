# p3-bicikelj-MCreeper12731

## Potrebne knižnjice

Z naslednjim ukazom naložimo vse potrebne knižnjice za zagon programa.

```pip install numpy pandas scikit-learn matplotlib seaborn plotly torch shap```

## Skripte

V mapi `scripts` so prisotni programi, ki generirajo .csv datoteke. `holidays.py` generira vse slovenske dela proste dneve, `meteodata.py` pa pridobi vremenske podatke (temperatura, količina padavin, hitrost vetra) od 1. januarja 2022 do 19. maja 2025. Obe skripti se zažene brez dodatnih parametrov.

## Raziskovanje podatkov

V mapi `models.ipynb` je podrobnejša analiza podatkov, razlaga modelov in nekatere grafične predstavitve.

## Generiranje podatkov

Skripta `final.py` generira vse napovedi za tekmovalni strežnik. Zažene se jo brez dodatnih parametrov