# Comparando os algoritmos DQN, A2C e PPO

O objetivo deste projeto é comparar três algoritmos de reinforcement learning considerando alguns ambientes disponíveis na biblioteca [Gymnasium](https://gymnasium.farama.org/). Os algoritmos que serão comparados são [DQN](https://arxiv.org/abs/1312.5602), [A2C](https://arxiv.org/abs/1602.01783) e [PPO](https://arxiv.org/abs/1707.06347).

Os ambientes que serão utilizados na comparação são: 
* [Bipedal Walker](https://gymnasium.farama.org/environments/box2d/bipedal_walker/);
* [Car Racing](https://gymnasium.farama.org/environments/box2d/car_racing/), versão discreta e contínua;
* [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/), e;
* [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

Todas as implementações dos algoritmos citados acima serão feitas utilizando a biblioteca [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/).

## Matriz de comparação

[Documento com a matriz de comparação a ser executada neste projeto](m.pdf)

## Estrutura do repositório

Este repositório está estruturado da seguinte forma: 
* no diretório raiz estão todos os scripts que executam o treinamento, salvam os dados do treinamento e o modelo.
* o diretório **results** deve armazenar todos os arquivos CSV com os dados dos treinamentos.
* o diretório **models** deve armazenar todos os modelos gerados a partir do treinamento. 

## Comandos úteis

```bash
jupyter nbconvert --to html --no-input analise_curva_aprendizado.ipynb
mv analise_curva_aprendizado.html report/analise_curva_aprendizado.html
```
