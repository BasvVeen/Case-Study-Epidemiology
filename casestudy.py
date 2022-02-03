import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm


from osgeo import gdal, ogr, osr

import pcraster as pcr
import pcraster.framework as pcrfw

import campo

seed = 5
pcr.setrandomseed(seed)




class FoodEnvironment(pcrfw.DynamicModel):
    def __init__(self):
        pcrfw.DynamicModel.__init__(self)
        # Framework requires a clone
        # set a dummy clone
        pcr.setclone(10, 20, 10, 0, 0)

    ##########################
    # differential equations #
    ##########################

    # first term, differential equation internal effects
    def diffEqTermOne(self, x, a, betaH, gammaH):
        return -((betaH / (1.0 + campo.exp(-gammaH * (x - a)))) - (betaH / 2.0))

    # second term, differential equation food outlet effects
    def diffEqTermTwo(self, y, a, betaS, gammaS, s):
        return ((betaS / (1.0 + campo.exp(-gammaS * (y - a + 0.7*s)))) - (betaS / 2.0))


    def initial(self):
        init_start = datetime.datetime.now()
        self.foodenv = campo.Campo()

        education = pd.read_csv('education_1000.csv', header = None)
        education = list(education[0])
        ##############
        # Households #
        ##############

        # create households phenomenon
        self.hh = self.foodenv.add_phenomenon('hh')
        self.hh.add_property_set('fd', '1000_households.csv')

        # set default propensity parameter
        self.hh.fd.lower = -0.0001
        self.hh.fd.upper = 0.0001
        self.hh.fd.a = campo.uniform(self.hh.fd.lower, self.hh.fd.upper, seed)

        # Create household education level
        # self.hh.fd.lower_ed = -0.05
        # self.hh.fd.upper_ed = 0.05
        # self.hh.fd.edu = campo.uniform(self.hh.fd.lower_ed, self.hh.fd.upper_ed, seed)

        self.hh.fd.edu = 0

        for i in tqdm(range(len(education))):
            self.hh.fd.edu.values()[i] = np.array([education[i]])

        # Create sport propensity levels
        self.hh.fd.lowers = -0.02
        self.hh.fd.uppers = 0.02
        self.hh.fd.s = campo.uniform(self.hh.fd.lowers, self.hh.fd.uppers, seed)

        # set betaH parameter
        self.hh.fd.betaH = 8.0

        # set gammaH parameter
        self.hh.fd.gammaH = 0.8
        self.hh.fd.resultingSlopeAtZero = (self.hh.fd.gammaH * self.hh.fd.betaH) / 4.0

        # set betaS parameter
        proportionOne = 0.7
        self.hh.fd.betaS = proportionOne * self.hh.fd.betaH

        print('Na betaS')

        # set gammaS parameter
        proportionTwo = 4.0
        self.hh.fd.gammaS = ((4 * self.hh.fd.resultingSlopeAtZero) / self.hh.fd.betaS) * proportionTwo

        # set initial propensity of households
        self.hh.fd.lower = -2
        self.hh.fd.upper = 2
        self.hh.fd.x = campo.uniform(self.hh.fd.lower, self.hh.fd.upper, seed)

        self.hh.fd.x = self.hh.fd.x + self.hh.fd.edu*0.2  #standaard waarde is 0.2 voor edu

        print('Na initial propensity')

        # add the surroundings property set
        self.hh.add_property_set('sur', '1000_houses_surrounding.csv')

        print('Na toevoegen surroundings')

        # calculate distance away from center
        # assign location of shop to property in surroundings property set
        self.hh.sur.start_locations = campo.feature_to_raster(self.hh.sur, self.hh.fd)
        # set some parameters for distance calculation
        self.hh.sur.initial_friction = 0
        self.hh.sur.friction = 1
        # calculate the distance
        self.hh.sur.distance = campo.spread(self.hh.sur.start_locations, self.hh.sur.initial_friction, self.hh.sur.friction)

        print('Na distance hh')

        # calculate the weight for averaging propensity of households in surroundings
        # calculate a zone of less than maxdistance (m) away from foodstore
        self.hh.sur.area = self.hh.sur.distance <= 1000
        # set value to assign outside zone and inside zone
        low = 0.000001
        high = 1.0
        self.hh.sur.low = low
        self.hh.sur.high = high
        # calculate the weight
        self.hh.sur.weight = campo.where(self.hh.sur.area, self.hh.sur.high, self.hh.sur.low)


        # technical detail
        self.hh.set_epsg(28992)


        ##############
        # Foodstores #
        ##############

        # create foodstores phenomenon
        self.fs = self.foodenv.add_phenomenon('fs')

        # add the frontdoor property set
        self.fs.add_property_set('fd', 'store_locations.csv')

        # add the surroundings property set
        self.fs.add_property_set('sur', 'foodstore_sur.csv')

        # calculate distance away from center
        # assign location of shop to property in surroundings property set
        self.fs.sur.start_locations = campo.feature_to_raster(self.fs.sur, self.fs.fd)
        # set some parameters for distance calculation
        self.fs.sur.initial_friction = 0
        self.fs.sur.friction = 1
        # calculate the distance
        self.fs.sur.distance = campo.spread(self.fs.sur.start_locations, self.fs.sur.initial_friction, self.fs.sur.friction)

        # calculate the weight for averaging propensity of households in surroundings
        # calculate a zone of less than 250 m away from foodstore
        self.fs.sur.area = self.fs.sur.distance <= 1000
        # set value to assign outside zone and inside zone
        self.fs.sur.high = high
        self.fs.sur.low = low
        # calculate the weight
        self.fs.sur.weight = campo.where(self.fs.sur.area, self.fs.sur.high, self.fs.sur.low)

        # technical detail
        self.fs.set_epsg(28992)

        ####################
        # Sport facilities #
        ####################

        # Create sporting facilities phenomenon
        self.sport = self.foodenv.add_phenomenon('sports')
        self.sport.add_property_set('fd', 'sport_locations.csv')

        self.sport.add_property_set('sur', 'sport_surrounding.csv')

        self.sport.set_epsg(28992)

        # Create quality measure of gyms

        # self.sport.fd.lowerq = -0.2
        # self.sport.fd.upperq = 0.2
        # self.sport.fd.q = campo.uniform(self.sport.fd.lowerq, self.sport.fd.upperq, seed)

        self.sport.fd.q = 0

        weights = pd.read_csv('weight112.csv', header = None)

        weights = list(weights.iloc[:,0])

        lower, upper = -0.2, 0.2
        w_norm = [lower + (upper - lower) * x for x in weights]

        for i in range(len(w_norm)):
            self.sport.fd.q.values()[i] = np.array([w_norm[i]])

        # Calculate average quality of sport facilities within buffer radius

        self.hh.fd.qual = campo.focal_agents(self.hh.fd, self.hh.sur.weight, self.sport.fd.q, fail=False)

        # Calculate total sport propensity

        self.hh.fd.s = self.hh.fd.s + self.hh.fd.edu*0.2 + self.hh.fd.qual

        # calculate propensity of surrounding households for each foodstore as starting point for dynamic part
        self.fs.fd.y = campo.focal_agents(self.fs.fd, self.fs.sur.weight, self.hh.fd.x, fail=False)

        # set the duration (years) of one time step
        self.timestep = 0.333333

        # create the output lue data set
        self.foodenv.create_dataset("food_environment.lue")

        # create real time settings for lue
        date = datetime.date(2000, 1, 2)
        time = datetime.time(12, 34)
        start = datetime.datetime.combine(date, time)
        unit = campo.TimeUnit.month
        stepsize = 4
        self.foodenv.set_time(start, unit, stepsize, self.nrTimeSteps())

        # technical detail
        self.hh.fd.x.is_dynamic = True
        self.hh.fd.y = 0.0 # temporary value
        self.hh.fd.y.is_dynamic = True
        self.fs.fd.y.is_dynamic = True

        # write the lue dataset
        self.foodenv.write()

        # print the run duration
        end = datetime.datetime.now() - init_start
        print(f'init: {end}')

    def dynamic(self):
        start = datetime.datetime.now()

        # average store propensity in neighbourhood of houses
        self.hh.fd.y = campo.focal_agents(self.hh.fd, self.hh.sur.weight, self.fs.fd.y, fail = False)

        ## update household propensity
        self.hh.fd.x = self.hh.fd.x + self.timestep \
                       * (self.diffEqTermOne(self.hh.fd.x, self.hh.fd.a, self.hh.fd.betaH, self.hh.fd.gammaH) \
                       + self.diffEqTermTwo(self.hh.fd.y, self.hh.fd.a, self.hh.fd.betaS, self.hh.fd.gammaS, self.hh.fd.s))

        #  average house propensity in neighbourhood of stores
        self.fs.fd.y = campo.focal_agents(self.fs.fd, self.fs.sur.weight, self.hh.fd.x, fail = False)

        # print run duration info
        self.foodenv.write(self.currentTimeStep())
        end = datetime.datetime.now() - start
        print(f'ts:  {end}  write')


if __name__ == '__main__':
  timesteps = 12
  myModel = FoodEnvironment()
  dynFrw = pcrfw.DynamicFramework(myModel, timesteps)
  dynFrw.run()
