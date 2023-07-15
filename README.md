It’s well-known in HR that recruiting new employees is substantially more expensive than retaining existing talent. Employees who depart take with them valuable experience and knowledge from your organisation. According to Forbes, the cost of an entry-level position turning over is estimated at 50% of that employee’s salary. For mid-level employees, it’s estimated at 125% of salary, and for senior executives, a whopping 200% of salary.

We’ll train some machine learning models in a Jupyter notebook using data about an employee’s position, happiness, performance, workload and tenure to predict whether they’re going to stay or leave.

Our target variable’s categorical, hence the ML task is classification. (For a numerical target, the task becomes regression.)

We’ll use a dataset from elitedatascience.com that simulates a large company with 14,249 past and present employees. There are 10 columns.


Snapshot of the original dataset.
The steps are:

EDA & data-processing: explore, visualise and clean the data.
Feature engineering: leverage domain expertise and create new features.
Model training: we’ll train and tune some tried-and-true classification algorithms, such as logistic regression, random forests and gradient-boosted trees.
Performance evaluation: we’ll look at a range of scores including F1 and AUROC.
Deployment: batch-run or get some data engineers / ML engineers to build an automated pipeline?
Ideally, the company will run the model on their current permanent employees to identify those at-risk. This is an example of machine learning providing actionable business insights.

New to AI or ML? Check out my explainer articles here and here.

Join Medium here and gain unlimited access to the best data science articles on the internet.

1. Data exploration and processing
Exploratory data analysis (EDA) helps us understand the data and provides ideas and insights for data cleaning and feature engineering. Data cleaning prepares the data for our algorithms while feature engineering is the magic sauce that will really help our algorithms draw out the underlying patterns from the dataset. Remember:

Better data always beats fancier algorithms!

We start by loading some standard data science Python packages into JupyterLab.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,
                             GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score,
                            f1_score, roc_curve, roc_auc_score
import pickle
Import the dataset:

df = pd.read_csv('employee_data.csv')
Here’s a snapshot of our dataframe again. The shape is (14,249, 10).


Snapshot of the original dataset.
The target variable is status. This categorical variable takes the value Employed or Left.

There are 25 columns/features:

department
salary
satisfaction, filed_complaint — proxies for happiness
last_evaluation, recently_promoted — proxies for performance
avg_monthly_hrs, n_projects — proxies for workload
tenure — proxy for experience
1.1 Numerical features
Let’s plot some quick histograms to get an idea of the distributions of our numerical features.

df.hist(figsize=(10,10), xrot=-45)

Things to do to our numerical features to ensure the data will play nice with our algorithms:

Convert the NaN’s in filed_complaint and recently_promoted to 0. They were incorrectly labelled.
Create an indicator variable for the missing data in the last_evaluation feature, before converting the NaN’s to zero.
df.filed_complaint.fillna(0, inplace=True)
df.recently_promoted.fillna(0, inplace=True)
df['last_evaluation_missing'] =         
df.last_evaluation.isnull().astype(int)
df.last_evaluation.fillna(0, inplace=True)
Here is a correlation heatmap for our numerical features.

sb.heatmap(df.corr(),
 annot=True,
 cmap=’RdBu_r’,
 vmin=-1,
 vmax=1)

1.2 Categorical features
Let’s plot some quick bar plots for our categorical features. Seaborn is great for this.

for feature in df.dtypes[df.dtypes=='object'].index:
    sb.countplot(data=df, y='{}'.format(features))



The biggest department is sales. Only a small proportion of employees are in the high salary bracket. And our dataset is imbalanced in that only a minority of employees have left the company, i.e. only a small proportion of our employees have status = Left. This has ramifications for the metrics we choose to evaluate our algorithms’ performances. We’ll talk more about this in the Results.

From a data-cleaning point of view, the IT and information_technology classes for the department feature should be merged together:

df.department.replace('information_technology', 'IT', inplace=True)
Moreover, HR only cares about permanent employees, so we should filter out the temp department:

df = df[df.department != 'temp']
Thus our department feature should look more like this:


Things to do to our categorical features to ensure the data will play nice with our algorithms:

Missing data for the department feature should be lumped into its own Missing class.
The department and salary categorical features should also be one-hot encoded.
The target variable status should be converted to binary.
df['department'].fillna('Missing', inplace=True)
df = pd.get_dummies(df, columns=['department', 'salary'])
df['status'] = pd.get_dummies(df.status).Left
1.3 Segmentations
We can draw further insights by segmenting numerical features against categorical ones. Let’s start off with some univariate segmentations.

Specifically, we’re going to segment numerical features representing happiness, performance, workload and experience by our categorical target variable status.

Segment satisfaction by status:

sb.violinplot(y='status', x='satisfaction', data=df)

An insight is that a number of churned employees were very satisfied with their jobs.

Segment last_evaluation status:

sb.violinplot(y='status', x='last_evaluation', data=df)

An insight is a large number of churned employees were high performers. Perhaps they felt no further opportunities for growth by staying?

Segment avg_monthly_hrs and n_projects by status:

sb.violinplot(y='status', x='avg_monthly_hrs', data=df)
sb.violinplot(y='status', x='n_projects', data=df)


It appears that those who have churned tended to either have a fairly large workload or a fairly low workload. Do these represent burnt out and disengaged former employees?

Segment tenure by status:

sb.violinplot(y='status', x='tenure', data=df)

We note that employee churn suddenly during the 3rd year. Those who are still around after 6 years tend to stay.

2. Feature engineering
Check out the following bivariate segmentations that will motivate our feature engineering later on.

For each plot, we’re going to segment two numerical features (representing happiness, performance, workload or experience) by status. This might give us some clusters based on employee stereotypes.

Performance and happiness:


Whoops, the Employed workers make this graph hard to read. Let’s just display the Left workers only, as they’re the ones we’re really trying to understand.

sb.lmplot(x='satisfaction',
          y='last_evaluation',
          data=df[df.status=='Left'],
          fit_reg=False
         )

We have three clusters of churned employees:

Underperformers: last_evaluation < 0.6
Unhappy: satisfaction_level < 0.2
Overachievers: last_evaluation > 0.8 and satisfaction > 0.7
Workload and performance:

sb.lmplot(x='last_evaluation',
          y='avg_monthly_hrs',
          data=df[df.status=='Left'],
          fit_reg=False
         )

We have two clusters of churned employees:

Stars: avg_monthly_hrs > 215 and last_evaluation > 0.75
Slackers: avg_monthly_hrs < 165 and last_evaluation < 0.65
Workload and happiness:

sb.lmplot(x='satisfaction',
          y='avg_monthly_hrs',
          data=df[df.status=='Left'],
          fit_reg=False,
         )

We have three clusters of churned employees:

Workaholics: avg_monthly_hrs > 210 and satisfation > 0.7
Just-a-job: avg_monthly_hrs < 170
Overworked: avg_monthly_hrs > 225 and satisfaction < 0.2
Let’s engineer new features for these 8 ‘stereotypical’ clusters of employees:

df['underperformer'] = ((df.last_evaluation < 0.6) & (df.last_evaluation_missing==0)).astype(int)
df['unhappy'] = (df.satisfaction < 0.2).astype(int)
df['overachiever'] = ((df.last_evaluation > 0.8) & (df.satisfaction > 0.7)).astype(int)
df['stars'] = ((df.avg_monthly_hrs > 215) & (df.last_evaluation > 0.75)).astype(int)
df['slackers'] = ((df.avg_monthly_hrs < 165) & (df.last_evaluation < 0.65) & (df.last_evaluation_missing==0)).astype(int)
df['workaholic'] = ((df.avg_monthly_hrs > 210) & (df.satisfaction > 0.7)).astype(int)
df['justajob'] = (df.avg_monthly_hrs < 170).astype(int)
df['overworked'] = ((df.avg_monthly_hrs > 225) & (df.satisfaction < 0.2)).astype(int)
We can take a glance at the proportion of employees in each of these 8 groups.

df[['underperformer', 'unhappy', 'overachiever', 'stars', 
    'slackers', 'workaholic', 'justajob', 'overworked']].mean()
underperformer    0.285257
unhappy           0.092195
overachiever      0.177069
stars             0.241825
slackers          0.167686
workaholic        0.226685
justajob          0.339281
overworked        0.071581
34% of employees are just-a-job employees — non-inspired and just here for the weekly pay cheque — while only 7% are flat out overworked.

Analytical base table: The dataset after applying all of these data cleaning steps and feature engineering is our analytical base table. This is the data on which we train our models.

Our ABT has 14,068 employees and 31 columns — see below for a snippet. Recall our original dataset had 14,249 employees and just 10 columns!


Enjoying this story? Get an email when I post similar articles.

3. Modelling
We’re going to train four tried-and-true classification models:

logistic regressions (L1 and L2-regularised)
random forests
gradient-boosted trees
First, let’s split our analytical base table.

y = df.status
X = df.drop('status', axis=1)
We’ll then split into training and test sets. Our dataset is mildly imbalanced, so we’ll use stratified sampling to compensate.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=df.status)
We’ll set up a pipeline object to train. This will streamline our model training process.

pipelines = {
       'l1': make_pipeline(StandardScaler(), 
             LogisticRegression(penalty='l1', random_state=123)),
       'l2': make_pipeline(StandardScaler(), 
             LogisticRegression(penalty='l2', random_state=123)),
       'rf': make_pipeline(
             RandomForestClassifier(random_state=123)),
       'gb': make_pipeline(
             GradientBoostingClassifier(random_state=123))
            }
We also want to tune the hyperparameters for each algorithm. For logistic regression, the most impactful hyperparameter is the strength of the regularisation, C.

l1_hyperparameters = {'logisticregression__C' : [0.001, 0.005, 0.01, 
                       0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
                     }
l2_hyperparameters = {'logisticregression__C' : 
                       [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 
                        1, 5, 10, 50, 100, 500, 1000]
                     }
For our random forest, we’ll tune the number of estimators (n_estimators), the max number of features to consider during a split (max_features), and the min number of samples to be a leaf (min_samples_leaf).

rf_hyperparameters = {
    'randomforestclassifier__n_estimators' : [100, 200],
    'randomforestclassifier__max_features' : ['auto', 'sqrt', 0.33],
    'randomforestclassifier__min_samples_leaf' : [1, 3, 5, 10]
    }
For our gradient-boosted tree, we’ll tune the number of estimators (n_estimators), learning rate, and the maximum depth of each tree (max_depth).

gb_hyperparameters = {
    'gradientboostingclassifier__n_estimators' : [100, 200],
    'gradientboostingclassifier__learning_rate' : [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth' : [1, 3, 5]
    }
We’ll save these hyperparameters in a dictionary.

hyperparameters = {
    'l1' : l1_hyperparameters,
    'l2' : l2_hyperparameters,
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
    }
Finally, we’ll fit and tune our models. Using GridSearchCV we can train all of these models with cross-validation on all of our declared hyperparameters with just a few lines of code!

fitted_models = {}
for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, 
                         hyperparameters[name], 
                         cv=10, 
                         n_jobs=-1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
4. Evaluation
I’ve written a dedicated article on popular machine learning metrics, including the ones used below.

4.1 Performance scores
We’ll start by printing the cross-validation scores. This is the average performance across the 10 hold-out folds and is a way to get a reliable estimate of the model performance using only your training data.

for name, model in fitted_models.items():
    print(name, model.best_score_)
Output:
l1 0.9088324151412831
l2 0.9088324151412831
rf 0.9793851075173272
gb 0.975475386529234
Moving onto the test data, we’ll:

calculate accuracy;
print the confusion matrix and calculate precision, recall and F1-score;
display the ROC and calculate the AUROC score.
Accuracy measures the proportion of correctly labelled predictions, however it is an inappropriate metric for imbalanced datasets, e.g. email spam filtration (spam vs. not spam) and medical testing (sick vs. not sick). For instance, if our dataset only had 1% of employees satisfying target=Left, then a model that always predicts the employee is still working at the company would instantly score 99% accuracy. In these situations, precision or recall is more appropriate. Whichever you use often depends on whether you want to minimise Type 1 errors (False Positives) or Type 2 errors (False Negatives). For spam emails, Type 1 errors are worse (some spam is OK as long as you don’t accidentally filter out an important email!) while Type 2 errors are unacceptable for medical testing (telling someone they didn’t have cancer when they did is a disaster!). The F1-score gets you the best of both worlds by taking the weighted average of precision and recall.

The area under the ROC, known as the AUROC is another standard metric for classification problems. It’s an effective measurement of a classifier’s ability to distinguish between classes and separate signal from noise. This metric is also robust against imbalanced datasets.

Here is the code to generate these scores and plots:

for name, model in fitted_models.items():
    print('Results for:', name)
    
    # obtain predictions
    pred = fitted_models[name].predict(X_test)
    # confusion matrix
    cm = confusion_matrix(y_test, pred)
    print(cm)
    # accuracy score
    print('Accuracy:', accuracy_score(y_test, pred))
    
    # precision
    precision = cm[1][1]/(cm[0][1]+cm[1][1])
    print('Precision:', precision)
    
    # recall
    recall = cm[1][1]/(cm[1][0]+cm[1][1])
    print('Recall:', recall)
    
    # F1_score
    print('F1:', f1_score(y_test, pred))
    
    # obtain prediction probabilities
    pred = fitted_models[name].predict_proba(X_test)
    pred = [p[1] for p in pred]
    # plot ROC
    fpr, tpr, thresholds = roc_curve(y_test, pred) 
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.plot(fpr, tpr, label=name)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate (TPR) i.e. Recall')
    plt.xlabel('False Positive Rate (FPR)')
    plt.show()
    
    # AUROC score
    print('AUROC:', roc_auc_score(y_test, pred))
Logistic regression (L1-regularised):

Output:
[[2015  126]
 [ 111  562]]

Accuracy:  0.9157782515991472
Precision: 0.8168604651162791
Recall:    0.8350668647845468
F1:        0.8258633357825129
AUROC:     0.9423905869485105

Logistic regression (L2-regularised):

Output:
[[2014  127]
 [ 110  563]]

Accuracy:  0.9157782515991472
Precision: 0.8159420289855073
Recall:    0.836552748885587
F1:        0.8261188554658841
AUROC:     0.9423246556128734

Gradient-boosted tree:

Output:
[[2120   21]
 [  48  625]]

Accuracy:  0.9754797441364605
Precision: 0.9674922600619195
Recall:    0.9286775631500743
F1:        0.9476876421531464
AUROC:     0.9883547910913578

Random forest:

Output:
[[2129   12]
 [  45  628]]
Accuracy:  0.9797441364605544
Precision: 0.98125
Recall:    0.9331352154531947
F1:        0.9565879664889566
AUROC:     0.9916117990718256

The winning algorithm is the random forest with an AUROC of 99% and a F1-score of 96%. This algorithm has a 99% chance of distinguishing between a Left and Employed worker… pretty good!

Out of 2814 employees in the test set, the algorithm:

correctly classified 628 Left workers (True Positives) while getting 12 wrong (Type I errors), and
correctly classified 2129 Employed workers (True Negatives) while getting 45 wrong (Type II errors).
FYI, here are the hyperparameters of the winning random forest, tuned using GridSearchCV.

RandomForestClassifier(bootstrap=True, 
                       class_weight=None, 
                       criterion='gini',
                       max_depth=None, 
                       max_features=0.33, 
                       max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       min_impurity_split=None,
                       min_samples_leaf=1, 
                       min_samples_split=2,
                       min_weight_fraction_leaf=0, 
                       n_estimators=200,
                       n_jobs=None, 
                       oob_score=False, 
                       random_state=123,
                       verbose=0, 
                       warm_start=False
                      )
4.2 Feature importances
Consider the following code.

coef = winning_model.feature_importances_
ind = np.argsort(-coef)
for i in range(X_train.shape[1]):
    print("%d. %s (%f)" % (i + 1, X.columns[ind[i]], coef[ind[i]]))
x = range(X_train.shape[1])
y = coef[ind][:X_train.shape[1]]
plt.title("Feature importances")
ax = plt.subplot()
plt.barh(x, y, color='red')
ax.set_yticks(x)
ax.set_yticklabels(X.columns[ind])
plt.gca().invert_yaxis()
This will print a list of features ranked by importance and a corresponding bar plot.

Ranking of feature importance:
1. n_projects (0.201004)
2. satisfaction (0.178810)
3. tenure (0.169454)
4. avg_monthly_hrs (0.091827)
5. stars (0.074373)
6. overworked (0.068334)
7. last_evaluation (0.063630)
8. slackers (0.028261)
9. overachiever (0.027244)
10. workaholic (0.018925)
11. justajob (0.016831)
12. unhappy (0.016486)
13. underperformer (0.006015)
14. last_evaluation_missing (0.005084)
15. salary_low (0.004372)
16. filed_complaint (0.004254)
17. salary_high (0.003596)
18. department_engineering (0.003429)
19. department_sales (0.003158)
20. salary_medium (0.003122)
21. department_support (0.002655)
22. department_IT (0.001628)
23. department_finance (0.001389)
24. department_management (0.001239)
25. department_Missing (0.001168)
26. department_marketing (0.001011)
27. recently_promoted (0.000983)
28. department_product (0.000851)
29. department_admin (0.000568)
30. department_procurement (0.000296)

There are three particularly strong predictors for employee churn:

n_projects (workload)
satisfaction (happiness) and
tenure (experience).
Moreover, these two engineered features also ranked high on the feature importance:

stars (high happiness & workload), and
overworked (low happiness & high workload).
Interesting, but not entirely surprising. The stars might have left for better opportunities while the overworked left after burning out.

5. Deployment

Image by ThisisEngineering RAEng.
An executable version of this model (.pkl) can be saved from the Jupyter notebook.

with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)
HR could pre-process new employee data before feeding it into the trained model. This is called a batch-run.

Once the modelling is done, you need to sell the impact of your data science work.

This means communicating your findings and explaining the value of your models and insights.

Here, data storytelling is crucial to career success.

In a large organisation, they might want to deploy the model into an production environment by engaging with data engineers and machine learning engineers.

These specialists build an automated pipeline around our model, ensuring that fresh data can be pre-processed and predictions reported to HR on a regular basis.

The model would now be in a production environment — taken care of by a 24/7 operations team — that can serve and scale your model to thousands of consumers or more.
