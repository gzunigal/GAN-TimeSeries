# Titulo

Uso de Redes Generativas Adversarias para el análisis de series de tiempo

# Resumen y palabras clave

La predicción de series de tiempo es un problema el cual se ha estudiado por muchos años debido al impacto que puede tener en la economía y bienestar mundial. Es por esto que existe una necesidad de diseñar, evaluar más y mejores métodos para este problema de predicción. Las redes generativas adversarias parecen tener un excelente desempeño generando series de tiempo indiferenciables de series reales. En este estudio se evidencia la necesidad académica de evaluar el desempeño de redes generativas adversarias en la predicción de series de tiempo de distintos niveles de complejidad.

**Palabras clave**: GAN, forecast, time-series

# Introducción

## Contexto

“Una red neuronal es una máquina diseñada para modelar la forma en que el cerebro realiza una tarea en particular o función de interés. La red es usualmente implementada usando componentes electrónicos o simulada a través de software” (Haykin, 2008)
Las redes neuronales artificiales nacen como una abstracción del funcionamiento del cerebro humano en base a la interacción entre neuronas. Estas se construyen formando una red de elementos excitables interconectados en una arquitectura específica, y son entrenadas para realizar una función específica usando un conjunto de datos adecuado.

Las redes neuronales han sido efectivas para realizar gran número de tareas que son difíciles de resolver con algoritmos tradicionales. Los principales usos que se les dan en la actualidad son de clasificación, detección de patrones y predicción. Es este último uso, el que será tratado en el presente trabajo.

Por otro lado, la predicción de series de tiempo ha tenido un fuerte avance en la última década, impulsado principalmente por el área financiera y el mercado bursátil. En este contexto, las redes neuronales han sido una herramienta que ha dado excelentes resultados, debido a su facilidad identificando patrones en conjuntos de datos a simple vista caóticos.

## Motivación

El estudio de series de tiempo, tanto su caracterización, modelación, inferencia de valores faltantes y especialmente la predicción de valores futuros, es relevante más que nunca en campos tan variados como la medicina, finanzas, evaluación de proyectos, meteorología y procesos logísticos, entre otros.

El desarrollo de nuevas arquitecturas de redes neuronales ha aumentado la cantidad de herramientas que se pueden ocupar para enfrentar el problema de predicción de series de tiempo. Es por esto que surge la necesidad académica de estudiar el comportamiento de las nuevas arquitecturas en distintos escenarios, de forma de tener un marco de comparación entre los distintos enfoques, que permita comprender las fortalezas y debilidades de cada arquitectura y, eventualmente, facilitar la elección de la herramienta más adecuada para problemas reales.

## Impacto esperado

Se espera que los resultados de este trabajo de investigación, en especial el modelo desarrollado y su comparación con modelos tradicionales, sea un precedente importante a la hora de elegir el modelo adecuado para un conjunto de datos similares en características al que se presenta. Esto permitirá acercar los beneficios de la aplicación de modelos complejos a tomadores de decisiones y profesionales que no tengan el conocimiento ni tiempo suficiente para comparar modelos.

# Planteamiento del problema científico

## Revisión preliminar de la literatura

El estudio de series de tiempo se remonta a antes de la década de los 60, el cual se basaba fuertemente en modelos econométricos (Zellner, 1959). Luego con el avance de las estadísticas, se comenzó a utilizar modelos más complejos como los modelos autorregresivos de media móvil (ARMA) y su equivalente con entradas exógenas (ARMAX) (Box, Jenkins, Reinsel, & Ljung, 2015). Estos modelos estadísticos se convirtieron, desde ese entonces, en la principal herramienta para la predicción de series de tiempo, y son consideradas al dia de hoy, como línea base o punto de comparación para nuevos métodos de predicción. Estos modelos tienen la particularidad de funcionar muy bien en contextos lineales, pero el estudio de fenómenos no lineales requiere el uso de modelos no lineales como el Modelo Autorregresivo No Lineal (NAR) y el Modelo Autorregresivo No Lineal de Media Movil (NARMA).

El primer uso publicado de redes neuronales para predicción de series de tiempo data del año 1990, la cual ocupa una red prealimentada (feedforward neural network) de tres capas, como capa oculta de 10 neuronas, la cual predice la cantidad anual de manchas solares  (Li, Mehrotra, Mohan, & Ranka, 1990). Desde ese año, se han hecho variadas investigaciones utilizando redes neuronales para predecir diferentes fenómenos, utilizando distintos tipos de redes neuronales, como perceptrón multicapa, redes convolucionales y redes recurrentes (Di Persio & Honchar, 2016)

En el año 2014 se presentó una nueva arquitectura de redes neuronales llamada Generative Adversarial Networks (GAN)(Goodfellow et al., 2014). Esta arquitectura consta de dos redes, una que cumple la función de generar datos falsos desde ruido, otra que se encarga de determinar si los datos entregados son reales o generados por la red generadora.

Estas redes se han aplicado a diferentes áreas de estudio, enfocándose principalmente en generación de imágenes y video, debido a su característica de generar datos faltos indistinguibles a datos reales. Desde el 2018, se ha estudiado la utilización de GANs en la predicción de series de tiempo sobre el precio del petróleo (Luo et al., 2018), series de tiempo multi-paso (Koesdwiady, Khatib, & Karray, 2018), el mercado agrícola (Chen, 2018), y acciones (Zhou, Pan, Hu, Tang, & Zhao, 2018).

## Brecha del conocimiento

A pesar de las primeros estudios con respecto a la utilización de GANs en la predicción de series de tiempo, es necesario analizar el desempeño de estas redes en distintos escenarios  usando como base de comparación, métodos más tradicionales con desempeño conocido, para cuantificar la mejora en distintos escenarios de prueba.

## Preguntas de investigación

* Es factible usar GANs para la predicción de series de tiempo?
* Tendrá el uso de GANs para la prediccion de series de tiempo una ventaja con respecto a los métodos tradicionales como ARMA y ARMAX?
* Cómo se compara la mejoría de resultados del uso de GANs por sobre ARMA y ARMAX para la predicción de series de tiempo, con respecto a otras arquitecturas de redes neuronales?

## Hipótesis del trabajo

Los resultados del uso de redes generativas adversarias para la predicción de series de tiempo es son significativamente superiores a los obtenidos al usar métodos tradicionales como ARMA y ARMAX

# Objetivo general y específicos del proyecto

## Objetivo general

Estudiar el uso de GANs en la predicción de series de tiempo

## objetivos específicos

* Definir categorías de complejidad para series de tiempo, las cuales permitan la comparación de los métodos (GAN, ARMA) bajo diferentes condiciones.
* Desarrollar un modelo de GAN que permita predecir valores de una serie de tiempo.
* Comparar los resultados obtenidos con GAN con los obtenidos al aplicar ARMA y ARMAX.
* Determinar si existe una mejora estadísticamente significativa en el uso de GANs por sobre ARMA y ARMAX en la predicción de series de tiempo.

# Referencias

# Anexo 1: Glosario

MLP: Multi Layer Perceptron o Perceptrón Multicapa

GAN: Generative Adversarial Network o Redes Generativas Adversarias

ARMA: Auto Regressive Movin Average Model o Modelo autorregresivo de Media Móvil

ARMAX: Auto Regressive - Moving-Average Model with Exogenous Inputs o Modelo Autorregresivo de Media Móvil con Entradas Exógenas
