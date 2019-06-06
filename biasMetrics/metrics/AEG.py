class AEG:
    """Method for calculating the Average Equality Gap (AEG) for true positive 
    rates (TPR) from a subpopulation and the background population to assess model 
    bias. AEG scores allow a closer look into how a binary classification model 
    performs across any specified subpopulation in the dataset. It compares how 
    the difference between TPR for a subpopulation the background population across 
    all probability thresholds. A perfectly balanced model will have a score of 0, 
    indicating there is no difference in the TPR between the two populations. A 
    total imbalance in the model will result in a score of 0.5 or -0.5, depending 
    on the direction of the skew. In this case all scores are interpreted relative 
    to the subpopulation. Positive scores indicate the model skews towards the 
    subpopulation and negative scores indicate the model skews away from the 
    subpopulation. 

    
    Conceptually this is difference between the curve of the rates (x(t)) and the 
    line y = x (y(t)) calculated as the integral (0, 1) of x(t) - y(t). This class 
    makes use of a simplified closed-form solution using the Mann Whitney U test. 

    There are two different AEG metrics included in this class.

    Positive AEG: 
    Calculates the average distance between the TPRs for all members of the 
    subpopulation and background population in the target class (1). Positive 
    scores indicate a rightward shift in the subpopulation and a tendency for the 
    model to produce false positives. Negative scores indicate a leftward shift in 
    the subpopulation and a tendency for the model to produce false negatives.

    Negative AEG:
    Calculates the average distance between the TPRs for all members of the 
    subpopulation and background population in the non-target class (0). Positive 
    scores indicate a rightward shift in the subpopulation and a tendency for the 
    model to produce false positives. Negative scores indicate a leftward shift in 
    the subpopulation and a tendency for the model to produce false negatives.



    Read more about how to compare scores in "Nuanced Metrics for Measuring 
    Unintended Bias with Real Data for Text Classification" by Daniel Borkan, 
    Lucas Dixon, Jeffrey Sorensen, Nithum Thain, Lucy Vasserman.

    https://arxiv.org/abs/1903.04561

    Methods
    ----------
    score : Calculates positive and negative AEG scores for all given parameters 
            and returns a heat map with the scores for each subpopulation.
    """

    def __init__(self):
        import pandas as pd
        self.output_df = pd.DataFrame()
        
        
    def score(self, y_true, y_probs, subgroup_df, output=True):
        """Parameters
        ----------
        y_true : pandas Series, pandas DataFrame
            The true values for all observations.
        y_pred : pandas Series, pandas DataFrame
            The model's predicted values for all observations.
        subgroup_df : pandas DataFrame
            Dataframe of all subgroups to be compared. Each column should be a
            specific subgroup with 1 to indicating the observation is a part of
            the subgroup and 0 indicating it is not. There should be no other values
            besides 1 or 0 in the dataframe.
        output : boolean (default = True)
            If true returns a heatmap of the AEG scores.
        """

        import numpy as np
        import pandas as pd
        from scipy.stats import mannwhitneyu

        def calc_pos_aeg(parameter, df): 
            sub_probs = df[((df.target == 1) & (df[parameter] == 1))]['probs']
            back_probs = df[((df.target == 1) & (df[parameter] == 0))]['probs']
            pos_aeg = (.5 - (mannwhitneyu(sub_probs, back_probs)[0] / (len(sub_probs)*len(back_probs))))
            return round(pos_aeg, 2) 
        
        def calc_neg_aeg(parameter, df): 
            sub_probs = df[((df.target == 0) & (df[parameter] == 1))]['probs']
            back_probs = df[((df.target == 0) & (df[parameter] == 0))]['probs']
            neg_aeg = (.5 - (mannwhitneyu(sub_probs, back_probs)[0] / (len(sub_probs)*len(back_probs))))
            return round(neg_aeg, 2) 

        # ensure that the passed dataframe has an appropriate axis    
        subgroup_df.reset_index(drop=True, inplace=True)


        # ensure input true and prob values are formatted correctly
        if type(y_true) == pd.core.frame.DataFrame:
            y_true.columns = ['target']
            y_true.reset_index(drop=True, inplace=True)
        else:
            y_true = pd.DataFrame(y_true, columns=['target']).reset_index(drop=True)

        if type(y_probs) == pd.core.frame.DataFrame:
            y_probs.columns = ['probs']
            y_probs.reset_index(drop=True, inplace=True)
        else:
            y_probs = pd.DataFrame(y_probs, columns=['probs']).reset_index(drop=True)
            
        # combine all inputs into a DataFrame
        input_df = pd.concat([y_true, y_probs, subgroup_df], axis=1)

        # build dataframe and fill with ROC AUC metrics
        self.output_df = pd.DataFrame(index=subgroup_df.columns, columns=['Positive AEG', 'Negative AEG'])
        for col in subgroup_df.columns:
            self.output_df.loc[col] = [calc_pos_aeg(col, input_df), 
                                       calc_neg_aeg(col, input_df)]

        if output:
            import seaborn as sns
            sns.heatmap(self.output_df.astype('float32'), 
                        vmin=-.5,
                        vmax=.5,
                        cmap=sns.diverging_palette(10, 10, n=101),
                        annot = True,
                        linewidths=2
                       );

