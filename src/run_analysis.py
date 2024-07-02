from pathlib import Path

from matplotlib import pyplot
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator
from political_party_analysis.loader import DataLoader
from political_party_analysis.visualization import scatter_plot
# comment
if __name__ == "__main__":

    data_loader = DataLoader()
    # Data pre-processing step
    df = data_loader.preprocess_data()

    # Dimensionality reduction step
    dimension_reducer = DimensionalityReducer(df)
    reduced_dim_data , model =dimension_reducer.dim_reducer()
    ## Uncomment this snippet to plot dim reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    density_est = DensityEstimator(data=reduced_dim_data,dim_reducer=model,high_dim_feature_names=df.columns)
    density_est.distribution()
    sample_kernel = density_est.kernal_sampling()
    sample_high_dim = density_est.inverse_mapping()
    # Plot density estimation results here
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        sample_kernel,
        color="r",
        splot=splot,
        label="Density Estimated",
    )    
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()
    ##### YOUR CODE GOES HERE #####
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")

    # Plot finnish parties here
    ##### YOUR CODE GOES HERE #####

    print("Analysis Complete")
