# Entrenamiento de Agentes de Reinforcement Learning para Sushi Go!

## Introducción

Bienvenido a la aplicación de entrenamiento de agentes RL para Sushi Go!, desarrollado como parte de un Trabajo de Fin de Grado (TFG). Este programa te permite entrenar y evaluar agentes de Aprendizaje por Refuerzo (RL) utilizando varios algoritmos, como A2C, DQN, PPO, QR-DQN y TRPO. La aplicación proporciona múltiples funcionalidades, desde el entrenamiento de modelos RL hasta la evaluación de agentes basados en reglas y la aplicación de aprendizaje por transferencia en diferentes entornos de jugadores.

El objetivo del TFG es desarrollar agentes capaces de jugar al Sushi Go! utilizando diversas técnicas de aprendizaje por refuerzo. Dado que el juego involucra decisiones estratégicas complejas, se enseña a los agentes a tomar decisiones ajustándose a las acciones de otros jugadores. Para alcanzar este objetivo, se procede a través de una serie de pasos, que incluyen el desarrollo de un entorno Gymnasium personalizado, el empleo de algoritmos de Reinforcement Learning proporcionados por la biblioteca Stable-Baselines3, y el desarrollo de una interfaz gráfica mediante Pygame, que nos permite jugar contra varios agentes. Finalmente, se evalúa y compara el desempeño de los agentes, con un enfoque en la transferencia de conocimiento entre distintas versiones del juego.

## Funcionalidades

La aplicación incluye las siguientes funcionalidades:

1. [Verificar Validez del Entorno](#1-verificar-validez-del-entorno)
2. [Entrenar un Modelo (Cronometrado)](#2-entrenar-un-modelo-cronometrado)
3. [Entrenar Todos los Algoritmos (A2C, DQN, PPO, QR-DQN, TRPO)](#3-entrenar-todos-los-algoritmos-a2c-dqn-ppo-qr-dqn-trpo)
4. [Eliminar Todos los Agentes Entrenados](#4-eliminar-todos-los-agentes-entrenados)
5. [Aplicar Aprendizaje por Transferencia (Entre entornos con diferentes cantidades de jugadores)](#5-aplicar-aprendizaje-por-transferencia-entre-entornos-con-diferentes-cantidades-de-jugadores)
6. [Eliminar Todos los Agentes de Aprendizaje por Transferencia](#6-eliminar-todos-los-agentes-de-aprendizaje-por-transferencia)
7. [Probar Agentes RL](#7-probar-agentes-rl)
8. [Probar Agentes Basados en Reglas](#8-probar-agentes-basados-en-reglas)


## Instrucciones de Instalación

Para comenzar, deberás instalar varias dependencias necesarias para la aplicación. Sigue los pasos a continuación:

### 1. Clonar el Repositorio

Clona este proyecto desde tu repositorio o descarga el código fuente.

### 2. Instalar Dependencias

Asegúrate de tener instalado Python 3.7+ en tu sistema. Luego, instala las dependencias necesarias ejecutando los siguientes comandos en tu terminal:

```bash
pip install numpy gymnasium pygame stable-baselines3 sb3-contrib
```

### 3. Configuración Adicional

Asegúrate de tener la siguiente estructura de carpetas para los modelos entrenados:

```text
SushiGo/
│
└───Training/
    └───Model/
```

<!-- ```text
Proyecto/
│
└───Training/
    └───Models/
    └───All_Models/
    └───TransferLearning/
``` -->

- **`Training/Model/`**: Aquí es donde deben colocarse los modelos RL para ser probados y evaluados. Los modelos entrenados con las opciones 3 y 5 deben moverse manualmente a este directorio para ser probados con las opciones 7 y 8.
Adicionalmente, en esta carpeta se presentan los agentes entrenados durante 1.2M de episodios, o partidas, en los entornos de 2 y 5 jugadores para cada uno de los algoritmos mencionados anteriormente.


### 4. Ejecución de la Aplicación

Para ejecutar la aplicación, utiliza el siguiente comando en tu terminal o consola de Python:

```bash
python ./sushi_go_v7.py
```

Este comando iniciará el programa, donde podrás seleccionar las opciones mencionadas anteriormente para entrenar, probar o jugar con los agentes RL o basados en reglas.

## Flujo de Trabajo de la Aplicación

Al ejecutar la aplicación, te muestra un menú con diversas opciones numeradas (del 1 al 8). A continuación se detalla cada una de las funcionalidades de la aplicación:

##### 1. Verificar Validez del Entorno
   Verifica si el entorno personalizado de Sushi Go! está configurado correctamente y es compatible con la API de Gymnasium, mediante el verificador de entornos de Gymnasium `check_env(env)`.
   

##### 2. Entrenar un Modelo (Cronometrado) 
   Entrena un agente RL seleccionado en la carpeta `Training/Model/` y cronometra el tiempo que toma el proceso de entrenamiento. 
   El proceso incluye la selección de:
   - **Número de jugadores**: Definir cuántos jugadores habrá en el entrenamiento.
   - **Algoritmo**: Seleccionar entre diferentes algoritmos de RL: A2C, DQN, PPO, QR-DQN, y TRPO.
   - **Número de episodios**: Especificar cuántos juegos/episodios se ejecutarán en el entrenamiento.
   - **Frecuencia de guardado**: Definir cada cuanto guardar el modelo entrenado, mientras se desarrollan los episodios, para utilizar la versión más reciente del modelo como oponente(s).
   - **Posibilidad de guardar modelos intermedios**: Posibilidad de guardar versiones intermedias del modelo en proceso. Si se introduce `y` o `yes`, se pregunta cada cuantos episodios se quiere obtener una versión intermedia.
   

#### 3. Entrenar Todos los Algoritmos
   Esta opción entrena agentes utilizando todos los algoritmos (A2C, DQN, PPO, QR-DQN, TRPO) para el número de jugadores especificado y guarda los modelos entrenados en una carpeta llamada `Training/P{num_players}`. Cada algoritmo tendrá su propia subcarpeta. Para probar o utilizar estos modelos, deberás mover manualmente el modelo deseado a la carpeta `Training/Model/`. Al igual que en la opción 2, debes proporcionar detalles como:
   - **Número de jugadores**.
   - **Número de episodios**.
   - **Frecuencia de guardado** de modelos y modelos intermedios.


#### 4. Eliminar Todos los Agentes Entrenados
   Elimina todos los modelos entrenados generados con la opción 3, limpiando la carpeta `P{num_players}` dentro de `Training/`.
   

#### 5. Aplicar Aprendizaje por Transferencia (Entre entornos con diferentes cantidades de jugadores)  
   Aplica aprendizaje por transferencia entre entornos con diferentes cantidades de jugadores, guardando los modelos resultantes en la carpeta `TransferLearning/`. Cada algoritmo tendrá su propia subcarpeta. Al igual que con la opción 3, deberás mover manualmente el modelo deseado a la carpeta `Training/Model/` para utilizarlo.

   El proceso incluye, la selección de:
   - **Modelo**: Puedes elegir entre los modelos previamente entrenados, ubicados en la carpeta `Training/Model/`.
   - **Número de jugadores** en el nuevo entorno.
   - **Número de episodios** para entrenar en este nuevo entorno.
   - **Frecuencia de guardado** de modelos y modelos intermedios.
   

#### 6. Eliminar Todos los Agentes de Aprendizaje por Transferencia 
   Elimina todos los modelos generados mediante aprendizaje por transferencia, es decir, elimina la carpeta `TransferLearning/`.
   

#### 7. Probar Agentes RL 
   Permite probar los agentes RL almacenados en la carpeta `Training/Model/` para evaluar su desempeño en el entorno de Sushi Go! y también jugar como humano, interactuando con los agentes a través de una interfaz gráfica desarrollada con Pygame. Si no hay jugadores humanos, se puede ver el desarrollo completo de la partida, incluyendo las manos de todos los participantes. Para ello, debes especificar:
     - **Número de jugadores.**
     - **Modo de renderizado**:
         - `print`: Modo consola, donde se imprime el estado completo del juego, incluyendo las manos de los demás, para la comprobación de la correctitud de la lógica del juego.
         - `pygame`: Modo visual interactivo con gráficos. Cuando no hay jugadores humanos se pueden observar todas las manos.
     - **Modelo de RL** (para el Player0): de entre los disponibles en la carpeta `Training/Model/`.
     - **Modo de los jugadores oponentes:** 
         - `RLAgent`: Un agente entrenado con Aprendizaje por Refuerzo (en la carpeta `Training/Model/`).
         - `RuleAgent`: Un agente basado en reglas.
         - `random`: Un agente que juega aleatoriamente.
         - `human`: Un jugador humano.
         
   Notas:
      - El jugador 0 siempre será un modelo de RL. 
      - Cuando no hay jugadores humanos, puedes avanzar en el juego presionando la tecla espacio. Además, puedes visualizar las demás manos haciendo clic en las flechas o utilizando las teclas de flecha derecha y flecha izquierda para moverte entre ellas.
      - Cuando hay jugadores humanos se avanza mediante la tecla del espacio o haciendo click en el botón de Listo.
      - Una vez se muestren los resultados de la partida, deberás pulsar espacio para continuar.
      - Si deseas cerrar la aplicación en cualquier momento durante el juego, puedes presionar la X en la ventana de Pygame.
   

#### 8. Probar Agentes Basados en Reglas 
   Prueba agentes basados en reglas predefinidas y, como en la opción anterior, permite que jugadores humanos interactúen con los agentes utilizando la interfaz gráfica. También cuenta con el modo de renderizado por consola para visualizar la lógica interna del juego y observar detalles como las cartas del oponente.

Para salir del programa, ingresa cualquier otro valor cuando se te solicite.


## Documentación

Este repositorio no solo contiene el código necesario para entrenar y evaluar agentes de Aprendizaje por Refuerzo (RL) en el juego **Sushi Go!**, sino que también incluye documentación relevante que puede ser útil para comprender mejor el contexto teórico y técnico del proyecto.

#### 1. Resumen del Trabajo

En la carpeta `docs/`, encontrarás un **resumen del trabajo** que describe en detalle los pasos seguidos para desarrollar este proyecto, y los resultados obtenidos. 

Este resumen está pensado para ofrecer una visión general del proyecto y los resultados obtenidos durante el proceso de desarrollo.

#### 2. Resumen de Sutton & Barto (Reinforcement Learning: An Introduction)

Además, el repositorio incluye un **resumen de los conceptos clave de "Reinforcement Learning: An Introduction"** de Sutton & Barto, uno de los textos fundamentales en el campo del Aprendizaje por Refuerzo. Este resumen abarca los siguientes temas:

Este material es ideal para aquellos que buscan profundizar en los conceptos teóricos detrás del trabajo realizado y entender mejor los algoritmos que se han aplicado en el proyecto.
