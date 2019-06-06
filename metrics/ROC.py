class NuancedROC:
    """Method for calculating nuanced AUR ROC scores to assess model bias.
    Nuanced AUC ROC scores allow for a closer look into how a classification
    model performs across any specifed sub-population in the trainging set. 
    There are three different types of nuanced roc metrics included in this
    class.

    Subgroup (SG) ROC: 
    This calculates the AUC ROC score for only a specific subgroup of the 
    population. This value can be compared against the overall AUC ROC score
    for the entire population to see if the model underperforms or overperforms
    in classifying the subgroup in question.

    Background Positive Subgroup Negative (BPSN) ROC:
    This calculates the AUC ROC score for positive (relative to the target)
    members of the background (non-subgroup) population and negative members
    of the subgroup population. This value can be compared to see how the 
    model performs at differentiating between positive members on the background
    population and negative members of the subgroup population.  

    Background Negative Subgroup Positive (BNSP) ROC:
    This calculates the AUC ROC score for negative (relative to the target)
    members of the background (non-subgroup) population and positive members
    of the subgroup population. This value can be compared to see how the 
    model performs at differentiating between negative members on the background
    population and positive members of the subgroup population.  

    Read more about how to compare scores in "Nuanced Metrics for Measuring 
    Unintended Bias with Real Data for Text Classification" by Daniel Borkan, 
    Lucas Dixon, Jeffrey Sorensen, Nithum Thain, Lucy Vasserman.

    https://arxiv.org/abs/1903.04561

    Methods
    ----------
    score : Calculates nuanced roc scores for all given parameters and and returns 
            a heat map with the scores for each subpopulation.

    Attributes
    ----------
    
    mean_SG_roc : Returns the mean of the SG ROCs for all subgroups.
        
    mean_BPSN_roc : Returns the mean of the BPSN ROCs for all subgroups.
        
    mean_BNSP_roc : Returns the mean of the BNSP ROCs for all subgroups.
        
    mean_roc : Returns the weighted mean of the SG, BPSN, and BNSP scores
               for all specified subgroups. 
        
    summary : Prints out all the scores for each subgroup.
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
            besides 1 or 0 in the dataframe."""

        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score

        def calc_SG_roc(parameter, df):
            SG = df.loc[df[parameter] == 1]
            SG_roc = roc_auc_score(y_true=SG.target, y_score=SG['probs'])
            return SG_roc

        # define functions to calculate specific ROC AUC for subpopulations within the data
        def calc_BPSN_roc(parameter, df):
            BPSN = df[((df.target == 1) & (df[parameter] == 0)) | ((df.target == 0) & (df[parameter] == 1))]
            BPSN_roc = roc_auc_score(y_true=BPSN.target, y_score=BPSN['probs'])
            return BPSN_roc

        def calc_BNSP_roc(parameter, df):
            BNSP = df[((df.target == 0) & (df[parameter] == 0)) | ((df.target == 1) & (df[parameter] == 1))]
            BNSP_roc = roc_auc_score(y_true=BNSP.target, y_score=BNSP['probs'])
            return BNSP_roc

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
        self.output_df = pd.DataFrame(index=subgroup_df.columns, columns=['SG-ROC', 'BPSN-ROC', 'BNSP-ROC'])
        for col in subgroup_df.columns:
            self.output_df.loc[col] = [calc_SG_roc(col, input_df), 
                                       calc_BPSN_roc(col, input_df), 
                                       calc_BNSP_roc(col, input_df)]

        self.model_roc = roc_auc_score(y_true=y_true, y_score=y_probs)

        self.mean_SG_roc = self.output_df['SG-ROC'].mean()

        self.mean_BPSN_roc = self.output_df['BPSN-ROC'].mean()

        self.mean_BNSP_roc = self.output_df['BNSP-ROC'].mean()

        self.mean_bias_roc = np.mean([self.output_df['SG-ROC'].mean(), 
                                      self.output_df['BPSN-ROC'].mean(), 
                                      self.output_df['BNSP-ROC'].mean()])

        if output:
            import seaborn as sns
            print(f'Model ROC: {round(self.model_roc, 3)}')
            sns.heatmap(self.output_df.astype('float32'), 
                        center = self.model_roc,
                        cmap='RdYlGn',
                        annot = True,
                        linewidths=2
                       );


    def summary(self):
        print(f'Model ROC: {self.model_roc}')
        print()
        print(f'Mean Bias ROC: {self.mean_bias_roc}')
        print()
        print(f'Mean SG ROC: {self.mean_SG_roc}')
        print()
        print(f'Mean BPSN ROC: {self.mean_BPSN_roc}')
        print()
        print(f'Mean BNSP ROC: {self.mean_BNSP_roc}')   
        print()
        print(self.output_df)