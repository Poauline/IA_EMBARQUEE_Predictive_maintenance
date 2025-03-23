# IA_EMBARQUEE_Predictive_maintenance
 
## Introduction

Aujourd'hui, la maintenance prédictive est incoutournable dans les industries de pointe. L'objectif est d'éviter des pannes impromptues et non anticipées qui retaderaient la production avec des conséquences économiques. L'idée de la maintenance préventive est d'analyser le comportement d'une machine pour déterminer une panne potentielle et agir en conséquence. Pour effectuer la maintenance préventive plusieurs approches existent. Tout d'abord, on peut simuler le comportement méchanique et électrique de la machine pour anticiper des pannes, cette technique est très complexe à mettre en place et oblige à des approximations. Une autre approche est une méthode data-driven. On récolte des données sur les machines et on interprète ces données passées pour identifier des pannes éventuelles dans le présent. 
C'est cette seconde méthode que nous allons mettre en place. Nous souhaitons créer un modèle d'intelligence artificielle pour identifier une panne potentielle via des paramètres de la machine. L'idée est de simuler un système mis en place dans une usine de fabrication de semi-conducteurs. Pour faire cela il est possible de faire remonter toutes les informations des capteurs à un serveur centralisé pour les interpréter ce qui oblige à avoir un stockage énorme et des connexions réseau. Sinon, on peut faire de l'IA en local sur la machine avec un RNN de petite taille utilisant en direct les données de la machine étudiée. Avec cette seconde méthode on peut ensuite mettre en place une architecture féderée en faisant remonter les informations à un serveur qui réentraîne un réseau de neurone en prenant en compte les nouvelles données des machines afin d'améliorer la précision du système.
Nous souhaitons donc d'intégrer en embarqué sur une carte STM32L152CC76 un réseau de neurone. Pour cela nous nous baserons sur un dataset labellisé qui va nous indiquer le bon fonctionnement ou non de la machine et le type de pannes pour différentes situations.
Puisque nous n'avons pas de machine de fabrication de semi-conducteurs sous la main nous simulerons ensuite l'envoi de données à la carte STM32 via un programme Python.
Dans ce README nous allons détailler toutes les étapes de conception du système de maintenance prédictive via une intelligence artificielle embarquée et présenter nos résultats.

## Conception du modèle d'intelligence artificielle pour la maintenance prédictive

### Les données d'entrée
Pour ce projet nous disposons d'un Dataset de données d'entrée dont on peut voir les premières lignes ci-dessous.
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
      <th>UDI</th>
      <th>Product ID</th>
      <th>Type</th>
      <th>Air temperature [K]</th>
      <th>Process temperature [K]</th>
      <th>Rotational speed [rpm]</th>
      <th>Torque [Nm]</th>
      <th>Tool wear [min]</th>
      <th>Machine failure</th>
      <th>TWF</th>
      <th>HDF</th>
      <th>PWF</th>
      <th>OSF</th>
      <th>RNF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>M14860</td>
      <td>M</td>
      <td>298.1</td>
      <td>308.6</td>
      <td>1551</td>
      <td>42.8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>L47181</td>
      <td>L</td>
      <td>298.2</td>
      <td>308.7</td>
      <td>1408</td>
      <td>46.3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>L47182</td>
      <td>L</td>
      <td>298.1</td>
      <td>308.5</td>
      <td>1498</td>
      <td>49.4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>L47183</td>
      <td>L</td>
      <td>298.2</td>
      <td>308.6</td>
      <td>1433</td>
      <td>39.5</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>L47184</td>
      <td>L</td>
      <td>298.2</td>
      <td>308.7</td>
      <td>1408</td>
      <td>40.0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

Les entrées sont : 
- UDI : l'identifiant
- Product ID : le produit
- Product type : le type de produit
- Air temperature : la température de l'air dans la machine
- Process temperature : la température interne
- Rotational speed : la vitesse de rotation
- Torque : le couple
- Tool wear : usure de l'outil (nombre de minutes d'utilisation)
- Machine failure : un booléen indiquand si on a une panne

Nous avons donc des données caractérisant les machines en fonctionnement et en panne que nous souhaitons utiliser pour prédire des pannes.
Avant toute chose il est important de constater que la très grande majorité des données correspond à des machines fonctionnant correctement. Les pannes correspondent à uniquement 3,39% des données (339 données). Cela va donc compliquer l'entraînement du modèle.
Nous pouvons observer la distribution des données sur le graphique ci-dessous.

![alt text](image.png)

Nous récupérons également le nombre de pannes de chaque type que nous traçons sur le diagramme ci-dessous. Notons que les pannes non définies (RNF) ne sont pas comptabilisées comme des pannes dans le Dataset (sauf pour une donnée, probablement dûe à un Dataset imparfait).

![alt text](image-1.png)

Il est important de garder en tête que nous souhaitons prédire un type de panne. Les pannes aléatoires sont par définition imprévisibles. Elles sont donc exclues du Dataset utilisé pour la conception du modèle.

Nous avons comme sorties :
- machine failure (0 ou 1)
- le type de failure (TWF, HDF, PWF, OSF, RNF qui sont à 0 ou 1)

Nous excluons donc RNF ainsi que les No Specific Failure qui sont des défauts dont la cause n'est pas identifiée.

Nous souhaitons créer une classe No Failure qui correspondra au cas où il n'y a pas de défaut. Nous créons donc une nouvelle colonne dans le Dataset avec :
**dataset['No Failure'] = 1 - dataset["Machine failure"]**

Ensuite, pour les entrées, nous conservons uniquement les entrées caractérisant le comportement de la machine, nous enlevons donc les 3 premières colonnes (l'identifiant et des informations sur le produit en cours de traitement).
### Conception du réseau de neurone
#### Les prédictions souhaitées

Notre objectif est de prédire l'appartenance à une catégorie de Failure (ou indiquer l'absence de risque) en fonction des données d'entrée.

Nous souhaitons donc mettre en place un réseau de neurone multi-classe. Il devra prédire l'appartenance à une classe et chaque élement pourra être associé à uniquement une classe.

Les classes de sortie sont : 
*   **0:NF**: No Failure
*   **1: TWF**: Tool Wear Failure
*   **2: HDF**: Heat Dissipation Failure
*   **3: PWF**: Power Failure
*   **4: OSF**: Overstrain Failure

#### Equilibrage du Dataset
Comme vu sur le graphique en première partie de ce document, notre Dataset n'est pas du tout équilibré avec 97,61% des données correspondant à la classe 1 (No Failure).

Conscients du problème, nous essayons tout de même d'entraîner un modèle avec ce Dataset non équilibré.
Nous voyons que, que l'on utilise un modèle très performant ou alors très simple, nous arrivons rapidement à 97% d'Accuracy. On peut déjà se douter qu'il y a un problème puisque 97% est la proportion des données dans la classe No Failure.
En traçant la matrice de confusion nous voyons bien que le modèle va constamment prédire la classe 0 puisqu'elle représente la grande majorité des cas.
![alt text](image-2.png)

Nous avons donc un modèle d'apprentissage automatique qui n'apprend pas et donne quasi systèmatiquement la même classe de sortie.

Nous allons donc balancer le Dataset. Pour cela nous disposons de plusieurs méthodes mais nous choisissons le SMOTE (Synthetic Minority Over-sampling Technique) qui consiste à génerer des nouvelles données à partir de celles existantes. Ainsi, en se mettant en mode Automatique, toutes les classes vont être équilibrées avec un nombre de données par classe correspondant au maximum de données dans une classe. 
De nouvelles données sont générées en interpolant linéairement entre un exemple sélectionné et l’un de ses voisins proches. Il ne s'agit pas d'un simple sur-échantillonage, on évite donc la duplication des données. Cependant, comme vu en cours. Cela peut introduire du bruit si les données des classes minoritaires sont trop dispersées ce qui ne semble pas être le cas ici.
Par exemple, les 115 valeurs de TWF vont être extrapolées pour obtenir 9670 valeurs (le nombre de No Failure).

Nous obtenons donc un Dataset avec 9670 valeurs pour les 5 classes et qui n'a donc plus de classe sur-représentée.

#### Architecture du modèle
Nous séparons les données du Dataset en 2 catégories : les données d'entraînement et les données de test. Nous utilisons 80% du Dataset pour l'entraînement et 20% pour le test. Nous ne prenons pas de données de validation.

Après avoir testé un CNN (Convolutionnal Neural Network) nous n'avons pas obtenu de résultats satisfaisants. En effet même avec de nombreuses couches, l'Accuracy restait de l'ordre de 30% ce qui correspond quasiment à du hasard avec 5 classes.

Nous nous sommes donc orientés vers un MLP (Multilayer Perceptron).
La couche d'entrée a pour dimension la dimension des données d'entrées donc 5.
En s'inspirant des projets déjà réalisés avec des MLP nous mettons 3 couches cachées de 32 à 64 neurones avec comme fonction d'activation Relu.
Enfin, nous avons un problème de classsification multi-classe. Nous utilisons donc une couche de sortie à 5 neurones (le nombre de classes en sortie) avec la fonction d'activation SoftMax.
Softmax va donner une probabilité d'appartenance à chacune des classes entre 0 et 1.

Pour connaître la classe la plus probable nous prenons celle avec la plus grande probabilité?