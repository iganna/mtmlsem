"""
Main class with a model
"""

from semopy import Model as semopyModel

from add_snps import *
from dataset import *
from lat_struct import *
from optimisation import *

from func_util import *

class mtmlModel:

    dict_lat_struct = {'unconnect': get_structure_unconnect,
                        'connected': get_structure_connected,
                        'spruce': get_structure_picea,
                        'prior': get_structure_prior}

    def __init__(self,
                 data: Data,
                 model_desc=None,
                 phens_under_kinship=None,
                 random_effects=None,
                 n_cv=None,
                 snp_multi_sort=True,
                 lat_struct_type=list(dict_lat_struct.keys())[0],
                 opt_type='bayes'):
        """
        Constructor of class model
        :param data: data for the model
        :param n_cv: number of cross-validation folds
        :param model_desc: string descriprtion of the model, should be provided if model_sem is not
        :param snp_multi_sort: TODO
        :param phens_under_kinship: set of phenotypes, which are under the kinship influence
        :param random_effects: list of names of variables with random effects
        :param lat_struct_type: type of the latent structure consctuction
        :param opt_type: type of optimisation
        :param reff: list of names of random effect variables
        """


        self.dict_opt = {'ml_reg': self.opt_ml,
                    'bayes': self.opt_bayes}
        # --------------------------------------------------
        # Dataset and cross-validation dataset
        self.data = data
        if random_effects is not None:
            if not all([v in self.data.d_all.columns for v in random_effects]):
                raise ValueError('Wrong names of random effects: '
                                 'variables do not exist in the dataset')

            # Add random effects if they were not added to the dataset before
            for v in random_effects:
                if v not in self.data.r_eff.keys():
                    self.data.r_eff.update({v: REff(self.data.d_all[v])})

        self.random_effects = random_effects

        # --------------------------------------------------
        # Mod descriptions in a dictionary
        self.mods = self.set_mod_description(model_desc=model_desc)
        # Add kinship do the model description
        if phens_under_kinship is not None:

            # Add kinship if is was not added to the dataset before
            if Data.kinship_var_name not in self.data.r_eff.keys():
                self.data.r_eff.update({Data.kinship_var_name:
                                        self.data.estim_kinship()})

            # TODO add kinship for all endogenous variables
            if phens_under_kinship == 'all_endo':
                phens_under_kinship = self.data.d_phens.colimns.tolist()
            self.phens_under_kinship = phens_under_kinship

            # Add kinship to all models
            self.add_kinship_to_mods()

            self.random_effects += ['kinship']
        else:
            self.phens_under_kinship = []


        # --------------------------------------------------
        # Cross-validation datasets
        if n_cv is None:
            self.cv_data = None
        else:
            self.cv_data = CVset(dataset=self.data, n_cv=n_cv)

        # --------------------------------------------------
        # parameters to construct latent structure
        self.lat_struct_type = self.set_lat_struct(lat_struct_type)

        # Parameters to add SNPs
        self.snp_multi_sort = snp_multi_sort

        # Parameters to estimate model parameters
        self.opt_type = self.set_opt_type(opt_type)

        self.relations_prior = self.empty_relations()
        self.relations = self.empty_relations()
        self.params = self.empty_relations()

    def add_kinship_to_mods(self):
        for k, mod in self.mods.items():
            sem = Model(mod)
            for v in sem.vars['all']:
                if v not in self.phens_under_kinship:
                    continue
                mod = f'{mod}\n{v} ~ {Data.kinship_var_name}'
            self.mods[k] = mod

    def empty_relations(self):
        return DataFrame(columns=['lval', 'rval', 'Estimate', 'mod_name'])

    def run_pipeline(self):

        self.get_lat_struct(cv=True)
        self.add_snps()
        opt = self.dict_opt[self.opt_type]
        # res = opt()


    def add_snps(self, snp_pref=None):
        for k, mod in self.mods.items():
            self.mods[k] = add_snps(mod, self.data,
                                    self.snp_multi_sort, snp_pref=snp_pref)


    def opt_bayes(self, n_mcmc=1000):

        self.relations = self.empty_relations()
        chains = dict()
        for k, mod in self.mods.items():
            sem = semopyModel(mod)
            sem.fit(self.data.d_all)
            relations_prior = sem.inspect()
            relations_prior = relations_prior.loc[relations_prior['op'] == '~',:]
            self.relations_prior = pd.concat([self.relations_prior,
                                              relations_prior], axis=0)
            # print(relations_prior.loc[:, ['lval', 'op', 'rval', 'Estimate']])
            opt = OptBayes(relations=relations_prior, data=self.data,
                           random_effects=self.random_effects)
            # return  opt
            opt.optimize(n_mcmc=n_mcmc)
            relations_tmp = opt.inspect()
            relations_tmp['mod_name'] = k
            self.relations = pd.concat([self.relations, relations_tmp], axis=0)
            chains[k] = opt.mcmc

        # print(self.relations)
        self.mcmc = chains
        # return {'params': self.relations, 'mcmc': chains}

        return opt


    def opt_ml(self):
        """
        Optimize parameters of sem models with semopy
        :return:
        """
        self.sems = []
        for k, mod in self.mods.items():
            sem = semopyModel(mod)
            sem.fit(self.data.d_all)

            # TODO
            self.relations = sem.inspect()


    def get_lat_struct(self, struct_type=None,
                       cv=False, n_cv=10,
                       loading_cutoff = None,
                       echo=False,
                       remain_models=False):
        """
        Function to construct the latent structure of the model
        :type remain_models: object
        :param struct_type:
        :param cv: False - use default parameters for factors, True - define them with CV
        :param n_cv: if CV-dataset was not defined, then define it with with number
        :return:
        """
        # Load the type of latent structure to object
        self.set_lat_struct(lat_struct_type=struct_type)

        # Whether to identify hyper-parameters
        if cv:
            if self.cv_data is None:
                self.cv_data = CVset(dataset=self.data, n_cv=n_cv)
            loading_cutoff = get_loading_cutoff(self.cv_data, echo=echo)

        get_structure = self.dict_lat_struct[self.lat_struct_type]

        if remain_models:
            self.mods.update(get_structure(data=self.data, loading_cutoff=loading_cutoff))
        else:
            self.mods = get_structure(data=self.data, loading_cutoff=loading_cutoff)


    # ---------------------------------------------
    # Set functions with checks
    def set_mod_description(self,
                            model_file=None,
                            model_sem: semopyModel = None,
                            model_desc=None):
        """
        Set the model from file or from the semopy Model
        :param model_file:
        :param model_sem:
        :param model_desc:
        :return:
        """

        if (model_file is None) + (model_sem is None) + (model_desc is None) == 1:
            raise ValueError('Only one way to set a model should be used')

        if (model_file is None) + (model_sem is None) + (model_desc is None) == 3:
            print('SEM model was not defined')
            return dict()

        if model_sem is not None:
            self.sems = model_sem
            return self.sems

        if model_file is not None:
            check_file(model_file)
            with open(model_file, 'r') as f:
                model_desc = f.read()
                show(model_desc)

        # Check correspondence between data and model
        sem_tmp = semopyModel(model_desc)
        sem_tmp.load_data(self.data.d_all)

        return {'mod_init': model_desc}


    def set_lat_struct(self, lat_struct_type=None):
        """
        Set way to construct the latent structure
        :param lat_struct_type:
        :return:
        """
        if lat_struct_type is not None:
            self.lat_struct_type = lat_struct_type
            if self.lat_struct_type not in list(self.dict_lat_struct.keys()):
                raise ValueError(f'Way to construct the structure is not:'
                                 f' {lat_struct_type}')
            if (self.lat_struct_type is not None) and \
                    (lat_struct_type != self.lat_struct_type):
                print(f'Way to construct the structure was changed: '
                      f'{lat_struct_type}')
            self.lat_struct_type = lat_struct_type
        else:
            print(f'Type of the latent structure is {self.lat_struct_type}')

        return self.lat_struct_type


    def set_opt_type(self, opt_type=None):
        """

        :param opt_type:
        :return:
        """
        if opt_type is not None:
            self.opt_type = opt_type
            if self.opt_type not in list(self.dict_opt.keys()):
                raise ValueError(f'Way to optimise is not: {opt_type}')
            if (self.opt_type is not None) and \
                    (opt_type != self.opt_type):
                print(f'Way to optimise was changed: {opt_type}')
            self.opt_type = opt_type

        return self.opt_type

    def unnormalize(self):

        # correct for first indicators of latent variables ?
        first_ind = self.relations_prior.loc[self.relations_prior['Std. Err'] == '-', ['lval', 'rval']]
        if len(first_ind['rval'].tolist()) != len(first_ind['rval'].unique().tolist()):
            raise ValueError('Something is wrong in the model.....')

        vars_lat = first_ind['rval'].tolist()
        vars_obs = first_ind['lval'].tolist()

        lat_correction = dict()
        for v_obs, v_lat in zip(vars_obs, vars_lat):
            idx_tmp = (self.relations['lval'] == v_obs) & \
                      (self.relations['rval'] == v_lat)
            lat_correction[v_lat] = (self.relations.loc[idx_tmp, 'Estimate'].item() *
                                         self.data.s_phens[v_obs])





        self.params = self.relations.copy()
        for index in self.params.index:
            lval = self.params.loc[index, 'lval']
            rval = self.params.loc[index, 'rval']
            if lval in self.data.s_phens.index:
                lstd = self.data.s_phens[lval]
            elif lval in lat_correction.keys():
                lstd = lat_correction[lval]
            else:
                lstd = 1

            if rval in self.data.s_phens.index:
                rstd = self.data.s_phens[rval]
            elif rval in lat_correction.keys():
                rstd = lat_correction[rval]
            else:
                rstd = 1


            self.params.loc[index, 'Estimate'] *= (lstd / rstd)

        return self.params


    def show_mod(self):
        print('======== show models ========')
        for k, mod in self.mods.items():
            print(f'# Model {k}')
            show(mod)
        print('=============================')



