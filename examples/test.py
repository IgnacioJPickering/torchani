import torchani

aev_computer = torchani.AEVComputer.from_neurochem_resource('ani-1x_8x.info')

energy_shifter = torchani.EnergyShifter.from_neurochem_resource('ani-1x_8x.info')

chemical_symbols_to_ints = torchani.utils.ChemicalSymbolsToInts.from_neurochem_resource('ani-1x_8x.info')

species_converter = torchani.SpeciesConverter.from_neurochem_resource('ani-1x_8x.info')

ani_model = torchani.ANIModel.from_neurochem_resource('ani-1x_8x.info', model_index=0)

ensemble = torchani.Ensemble.from_neurochem_resource('ani-1x_8x.info')

builtin_model = torchani.models.BuiltinModel.from_neurochem_resource('ani-1x_8x.info', model_index=0)

builtin_ensemble = torchani.models.BuiltinEnsemble.from_neurochem_resource('ani-1x_8x.info')
