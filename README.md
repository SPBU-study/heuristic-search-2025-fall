# Алгоритмы сокращения пространства поиска на графе регулярной декомпозиции за счет симметрии и смежных техник (JPS и его модификации)

## Необходимо реализовать базовый алгоритм JPS  [Harabor and Grastien, 2014], [Rabin and Sturtevant, 2016], опционально реализовать его улучшение из статьи  [Harabor and Grastien, 2014]. Далее реализовать версию алгоритма для поиска путей на графах-сетках с различным типом (и стоимостью) клеток [Carlson et al, 2023]

- Команда 2 человека: Денисов Никита и Васильев Егор
- Все статьи находятся в папке [papers](papers/)

## [Презентация КТ №1](https://docs.google.com/presentation/d/12X5laVY5Llpda4JvgQLUZ1vzJgyGd83-QKXvEm5NzWQ/edit?slide=id.g3a19b4d6b3b_1_96#slide=id.g3a19b4d6b3b_1_96)

## Инструкция по запуску

```
git clone https://github.com/SPBU-study/heuristic-search-2025-fall.git
cd heuristic-search-2025-fall/JPS
python -m pathfinding.cli --map maps/weighted-map/Map12.map --start-x 47 --start-y 295 --goal-x 1047 --goal-y 1443 --algorithm astarw --visualize
```