```python
#KNN____________________________________
# Librerías para manejo de datos
 # librería Natural Language Toolkit, usada para trabajar con textos 
# Librerías para manejo de datos
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import pandas as pd
pd.set_option('display.max_columns', 25) # Número máximo de columnas a mostrar
pd.set_option('display.max_rows', 50) # Numero máximo de filas a mostar
import numpy as np
np.random.seed(3301)
import pandas as pd
# Para preparar los datos
from sklearn.preprocessing import LabelEncoder
# Para crear el arbol de decisión 
from sklearn.tree import DecisionTreeClassifier 
# Para usar KNN como clasificador
from sklearn.neighbors import KNeighborsClassifier
# Para realizar la separación del conjunto de aprendizaje en entrenamiento y test.
from sklearn.model_selection import train_test_split
# Para evaluar el modelo
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix
# Para búsqueda de hiperparámetros
from sklearn.model_selection import GridSearchCV
# Para la validación cruzada
from sklearn.model_selection import KFold 
#Librerías para la visualización
import matplotlib.pyplot as plt
# Seaborn
import seaborn as sns 
from sklearn import tree
pd.options.mode.chained_assignment = None  # default='warn'
import pickle


#Modelo
import sklearn
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#k fold validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold 

#Métricas
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix
```


```python
#Se cargan los datos mediante la libreria pandas y se asigna una variable para los datos
data=pd.read_csv('data-HQ.csv', sep=',', encoding = 'utf-8')
data_preprocesados =data.dropna()
# Ver los datos
display(data_preprocesados.sample(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>text</th>
      <th>class</th>
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>114306</th>
      <td>114339</td>
      <td>130468</td>
      <td>I am getting to the point of killing myself.He...</td>
      <td>suicide</td>
      <td>get point kil myselfhey everyon nam start feel...</td>
    </tr>
    <tr>
      <th>102931</th>
      <td>102962</td>
      <td>74050</td>
      <td>Why cannot everyone stop talking about the nea...</td>
      <td>suicide</td>
      <td>everyon stop talk near fut try kil myselfnev u...</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>194708</td>
      <td>344258</td>
      <td>I just heard someone be transphobic on the rad...</td>
      <td>non-suicide</td>
      <td>heard someon transphob radio tot angry fuck no...</td>
    </tr>
    <tr>
      <th>140241</th>
      <td>140287</td>
      <td>188653</td>
      <td>At the end of my history class everytime, I tu...</td>
      <td>non-suicide</td>
      <td>end hist class everytim turn snap potato filt ...</td>
    </tr>
    <tr>
      <th>123922</th>
      <td>123960</td>
      <td>184059</td>
      <td>MedicationsWhat medications is anyone else tak...</td>
      <td>suicide</td>
      <td>medicationswh med anyon els tak cury also sid ...</td>
    </tr>
    <tr>
      <th>48650</th>
      <td>48660</td>
      <td>186023</td>
      <td>Dudes be paying for seeing Pussy pics. When th...</td>
      <td>non-suicide</td>
      <td>dud pay see pussy pic us mir post brought hoe ...</td>
    </tr>
    <tr>
      <th>116684</th>
      <td>116718</td>
      <td>137973</td>
      <td>Waking upDoes anyone wake up in the morning an...</td>
      <td>suicide</td>
      <td>wak updo anyon wak morn disappoint wake updoes...</td>
    </tr>
    <tr>
      <th>1172</th>
      <td>1172</td>
      <td>108811</td>
      <td>everything i have ever wanted, since before i ...</td>
      <td>suicide</td>
      <td>everyth ev want sint remember know ev going di...</td>
    </tr>
    <tr>
      <th>124206</th>
      <td>124244</td>
      <td>300482</td>
      <td>you know that person who no one ever counts on...</td>
      <td>non-suicide</td>
      <td>know person on ev count anyth everyon know goi...</td>
    </tr>
    <tr>
      <th>28698</th>
      <td>28704</td>
      <td>54642</td>
      <td>Story time Like last year I met this really cu...</td>
      <td>non-suicide</td>
      <td>story tim lik last year met real cut girl lik ...</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Dimensiones de los datos
data_preprocesados.shape
```




    (195639, 5)




```python
display(data_preprocesados.head(5)) # Primeras Filas
# Imprimimos los diferentes tipos de las columnas
data_preprocesados.dtypes
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>text</th>
      <th>class</th>
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>173271</td>
      <td>i want to destroy myselffor once everything wa...</td>
      <td>suicide</td>
      <td>want destroy myselff everyth start feel okay c...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>336321</td>
      <td>I kind of got behind schedule with learning fo...</td>
      <td>non-suicide</td>
      <td>kind got behind schedule learn next week testw...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>256637</td>
      <td>I am just not sure anymoreFirst and foremost: ...</td>
      <td>suicide</td>
      <td>sur anymorefirst foremost brazil judg second n...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>303772</td>
      <td>please give me a reason to liveThats too much ...</td>
      <td>suicide</td>
      <td>pleas giv reason liveth much reason liv lik an...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>293747</td>
      <td>27f struggling to find meaning moving forwardI...</td>
      <td>suicide</td>
      <td>27f struggling find mean mov forward admit bit...</td>
    </tr>
  </tbody>
</table>
</div>





    Unnamed: 0       int64
    Unnamed: 0.1     int64
    text            object
    class           object
    words           object
    dtype: object




```python
#Separar datos
X_data, y_data = data_preprocesados['words'],data_preprocesados['class']
y_data = (y_data == 'suicide').astype(int)
y_data
```




    0         1
    1         0
    2         1
    3         1
    4         1
             ..
    195634    0
    195635    0
    195636    0
    195637    0
    195638    1
    Name: class, Length: 195639, dtype: int64




```python
# Primero, es fundamental saber que un vector es una "recopilación de celdas o casilla que contienen variables individuales"
# En este paso, se vectorizan los datos por conteo de palabras. 
count = CountVectorizer()
X_count = count.fit_transform(X_data)
# Se revisa que se siga teniendo el mismo numero de datos que se tenia originalmente. 
print(X_count.shape)
```

    (195639, 204752)



```python
#KNN____________________________________________

# Se borran las columnas de unnamed: 0 y Unnames: 0.1, ya que no afecta el algoritmo y no son relevantes en este caso. 

data_borrandoColumnas = data_preprocesados
data_borrandoColumnas = data_borrandoColumnas.drop(['Unnamed: 0'], axis=1)
data_borrandoColumnas = data_borrandoColumnas.drop(['Unnamed: 0.1'], axis=1)
display(data_borrandoColumnas.head(5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>class</th>
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i want to destroy myselffor once everything wa...</td>
      <td>suicide</td>
      <td>want destroy myselff everyth start feel okay c...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I kind of got behind schedule with learning fo...</td>
      <td>non-suicide</td>
      <td>kind got behind schedule learn next week testw...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I am just not sure anymoreFirst and foremost: ...</td>
      <td>suicide</td>
      <td>sur anymorefirst foremost brazil judg second n...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>please give me a reason to liveThats too much ...</td>
      <td>suicide</td>
      <td>pleas giv reason liveth much reason liv lik an...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27f struggling to find meaning moving forwardI...</td>
      <td>suicide</td>
      <td>27f struggling find mean mov forward admit bit...</td>
    </tr>
  </tbody>
</table>
</div>


Teniendo en cuenta que el objetivo es a partir de textos planos, si una persona está intentando, va a intentar o ha intentado suicidarse con el fin de prevenir 
estos acontecimientos y/o brindar la ayuda necesaria a la gente que lo necesita, entonces se puede concluir que la variable objetivo es la "class", la cual dice si es suicide o non-suicide el 
texto 


```python
Y = data_borrandoColumnas['class']
X =data_borrandoColumnas.drop(['class'], axis=1)
```


```python
# Al ya haber escogido la variable objetivo, ya se puede dividir los datos en entrenamiento y test
X_train, X_test, Y_train, Y_test = train_test_split(X_count, y_data, test_size=0.2, random_state=0)
# se muestra
display(X.head(5)) # Primeras Filas
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i want to destroy myselffor once everything wa...</td>
      <td>want destroy myselff everyth start feel okay c...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I kind of got behind schedule with learning fo...</td>
      <td>kind got behind schedule learn next week testw...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I am just not sure anymoreFirst and foremost: ...</td>
      <td>sur anymorefirst foremost brazil judg second n...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>please give me a reason to liveThats too much ...</td>
      <td>pleas giv reason liveth much reason liv lik an...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27f struggling to find meaning moving forwardI...</td>
      <td>27f struggling find mean mov forward admit bit...</td>
    </tr>
  </tbody>
</table>
</div>



```python
arrayTrain = X_train.toarray()
arrayPredict = X_test.toarray()
# Imprime los lengths del array. 
print(len(arrayTrain))
print(len(arrayPredict))
```


```python
neigh = KNeighborsClassifier(n_neighbors=3)
neigh = neigh.fit(X_train, Y_train)
y_pred = neigh.predict(X_test)

# Se genera la matriz de confusión
confusion_matrix(Y_test, y_pred)

# Mostrar reporte de clasificación
print(classification_report(Y_test, y_pred))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/qx/6jkv_lxd5cb2rzf5xgs7m70r0000gn/T/ipykernel_71570/2855496488.py in <module>
          1 # Se genera la matriz de confusión
    ----> 2 confusion_matrix(Y_test, y_pred)
    

    NameError: name 'y_pred' is not defined



```python
print('Exactitud: %.2f' % accuracy_score(Y_test, y_pred))
print("Recall: {}".format(recall_score(Y_test,y_pred)))
print("Precisión: {}".format(precision_score(Y_test,y_pred)))
print("Puntuación F1: {}".format(f1_score(Y_test,y_pred)))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/qx/6jkv_lxd5cb2rzf5xgs7m70r0000gn/T/ipykernel_71570/1059776287.py in <module>
    ----> 1 print('Exactitud: %.2f' % accuracy_score(Y_test, y_pred))
          2 print("Recall: {}".format(recall_score(Y_test,y_pred)))
          3 print("Precisión: {}".format(precision_score(Y_test,y_pred)))
          4 print("Puntuación F1: {}".format(f1_score(Y_test,y_pred)))


    NameError: name 'y_pred' is not defined



```python

```
