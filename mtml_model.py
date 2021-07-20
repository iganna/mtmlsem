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
                 data: Data = None,
                 n_cv=None,
                 model_sem: semopyModel = None,
                 model_file=None,
                 model_desc=None,  # TODO
                 snp_multi_sort=True,
                 kinship_flag=True,
                 lat_struct_type=list(dict_lat_struct.keys())[0],
                 opt_type='bayes'):

        self.dict_opt = {'ml_reg': self.opt_ml,
                    'bayes': self.opt_bayes}

        # Dataset and cross-validation dataset
        self.data = data

        # Mod descriptions in a dictionary
        self.mods = self.set_mod_description(model_sem=model_sem,
                                             model_file=model_file,
                                             model_desc=model_desc)
        # Cross-validation datasets
        if n_cv is None:
            self.cv_data = None
        else:
            self.cv_data = CVset(dataset=self.data, n_cv=n_cv)


        # Kinship
        self.kinship_flag = kinship_flag
        if (self.kinship_flag) and (data.d_kinship is None) and (data is not None):
            data.estim_kinship()

        # parameters to construct latent structure
        self.lat_struct_type = self.set_lat_struct(lat_struct_type)

        # Parameters to add SNPs
        self.snp_multi_sort = snp_multi_sort

        # Parameters to estimate model parameters
        self.opt_type = self.set_opt_type(opt_type)

        self.relations_prior = self.empty_relations()
        self.relations = self.empty_relations()
        self.params = self.empty_relations()


    def empty_relations(self):
        return DataFrame(columns=['lval', 'rval', 'Estimate', 'mod_name'])

    def run_pipeline(self):

        self.get_lat_struct(cv=True)
        self.add_snps()
        opt = self.dict_opt[self.opt_type]
        res = opt()


    def add_snps(self):
        for k, mod in self.mods.items():
            self.mods[k] = add_snps(mod, self.data, self.snp_multi_sort)


    def opt_bayes(self):

        self.relations = self.empty_relations()
        chains = dict()
        for k, mod in self.mods.items():
            sem = semopyModel(mod)
            sem.fit(self.data.d_all)
            relations_prior = sem.inspect()
            relations_prior = relations_prior.loc[relations_prior['op'] == '~',:]
            self.relations_prior = pd.concat([self.relations_prior,
                                              relations_prior], axis=0)
            print(relations_prior.loc[:, ['lval', 'op', 'rval', 'Estimate']])
            opt = OptBayes(relations=relations_prior, data=self.data)
            opt.optimize()
            relations_tmp = opt.inspect()
            relations_tmp['mod_name'] = k
            self.relations = pd.concat([self.relations, relations_tmp], axis=0)
            chains[k] = opt.mcmc

        print(self.relations)
        self.mcmc = chains
        return {'params': self.relations, 'mcmc': chains}


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

        if (model_file is None) + (model_sem is None) + (model_desc is None) != 2:
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
                raise ValueError(f'Way to construct the structure is not: {lat_struct_type}')
            if (self.lat_struct_type is not None) and \
                    (lat_struct_type != self.lat_struct_type):
                print(f'Way to construct the structure was changed: {lat_struct_type}')
            self.lat_struct_type = lat_struct_type
        else:
            print(f'Empty type of latent structure was not set. The type remains as{self.lat_struct_type}')

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
        self.params = self.relations.copy()
        for index in self.params.index:
            lval = self.params.loc[index, 'lval']
            rval = self.params.loc[index, 'rval']
            if lval in self.data.s_phens.index:
                lstd = self.data.s_phens[lval]
            else:
                lstd = 1

            if rval in self.data.s_phens.index:
                rstd = self.data.s_phens[rval]
            else:
                rstd = 1


            self.params.loc[index, 'Estimate'] *= (lstd / rstd)

        return self.params





