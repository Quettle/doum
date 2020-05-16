import MultiNEAT as NEAT
from MultiNEAT import GetGenomeList, ZipFitness, EvaluateGenomeList_Serial
import click
import vizdoom as vzd
import pickle
import skimage.color, skimage.transform
import numpy as np
import datetime

resolution = (30, 45)

game = vzd.DoomGame()
game.load_config("doum.cfg")
game.init()

# the simple 2D substrate with 3 input points, 2 hidden and 1 output for XOR

substrate = NEAT.Substrate([(-1, i)for i in range(1350)],
		                   [],
		                   [(1, 0),(1, 1),(1, 2)])

substrate.m_allow_input_hidden_links = False
substrate.m_allow_input_output_links = False
substrate.m_allow_hidden_hidden_links = True
substrate.m_allow_hidden_output_links = True
substrate.m_allow_output_hidden_links = False
substrate.m_allow_output_output_links = False
substrate.m_allow_looped_hidden_links = True
substrate.m_allow_looped_output_links = False
substrate.m_allow_input_hidden_links = True
substrate.m_allow_input_output_links = True
substrate.m_allow_hidden_output_links = True
substrate.m_allow_hidden_hidden_links = True
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID
substrate.m_with_distance = True
substrate.m_max_weight_and_bias = 8.0

print(substrate)

params = NEAT.Parameters()

params.PopulationSize = 60

params.DynamicCompatibility = True
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 10
params.SpeciesMaxStagnation = 100
params.OldAgeTreshold = 30
params.MinSpecies = 5
params.MaxSpecies = 10
params.RouletteWheelSelection = False

params.MutateRemLinkProb = 0.02
params.RecurrentProb = 0.01
params.OverallMutationRate = 0.15
params.MutateAddLinkProb = 0.08
params.MutateAddNeuronProb = 0.01
params.MutateWeightsProb = 0.90
params.MaxWeight = 8.0
params.WeightMutationMaxPower = 0.2
params.WeightReplacementMaxPower = 1.0

params.MutateActivationAProb = 0.0
params.ActivationAMutationMaxPower = 0.5
params.MinActivationA = 0.05
params.MaxActivationA = 6.0

params.MutateNeuronActivationTypeProb = 0.03

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = .05
params.ActivationFunction_Tanh_Prob = 0.5
params.ActivationFunction_TanhCubic_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 1.0
params.ActivationFunction_UnsignedStep_Prob = 0.0
params.ActivationFunction_SignedGauss_Prob = 1.0
params.ActivationFunction_UnsignedGauss_Prob = 0.0
params.ActivationFunction_Abs_Prob = 0.0
params.ActivationFunction_SignedSine_Prob = 1.0
params.ActivationFunction_UnsignedSine_Prob = 0.0
params.ActivationFunction_Linear_Prob = 1.0

params.MutateNeuronTraitsProb = 0
params.MutateLinkTraitsProb = 0

params.AllowLoops = True





def preprocess(img):
	img = skimage.transform.resize(img, resolution)
	img = img.astype(float)
	img=img.reshape(1,1350)

	return img[0]



def evaluate(genome):
	net = NEAT.NeuralNetwork()
	genome.BuildHyperNEATPhenotype(net, substrate)
	net.Flush()
	game.new_episode()
	while not game.is_episode_finished():
		state = game.get_state()
		net.Input(preprocess(state.screen_buffer))
		net.Activate()
		out=net.Output()

		r = game.make_action([i>0.5 for i in out])
	return game.get_total_reward()

@click.command()
@click.option("--n_generations", type=int, default=100)
def run(n_generations):
	print("game ok")
	g = NEAT.Genome(0,
                    1350,#substrate.GetMinCPPNInputs(),
                    0,
                    3,#substrate.GetMinCPPNOutputs(),
                    False,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    0,
                    params, 0,0)

	pop = NEAT.Population(g, params, True, 1.0, 0)
	print("pop ok")
	print(substrate.GetMinCPPNInputs())
	for generation in range(n_generations):
		genome_list = NEAT.GetGenomeList(pop)

		fitnesses = EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
		[genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

		print('Gen: %d Best: %3.5f' % (generation, max(fitnesses)))

		pop.Epoch()
		generations = generation
	
	pop.Save(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".pop")


if __name__ == "__main__":
	run()	

