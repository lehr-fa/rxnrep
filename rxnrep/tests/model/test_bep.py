import numpy as np
import scipy.stats
import torch

from rxnrep.model.bep import ActivationEnergyPredictor, LinearRegression


def test_activation_energy_predictor():
    # two clusters
    reaction_e = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] * 2
    activation_e = [1.1, 1.9, 3.1, 3.8, 5.2, 0.0, 0.0] * 2
    has_activation_e = [1, 1, 1, 1, 1, 0, 0] + [1, 1, 0, 0, 0, 0, 0]
    assignments = [0, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1]

    reaction_e = torch.as_tensor(reaction_e)
    activation_e = torch.as_tensor(activation_e)
    has_activation_e = torch.as_tensor(has_activation_e, dtype=torch.bool)
    assignments = torch.as_tensor(assignments)

    predictor = ActivationEnergyPredictor(reaction_e, activation_e, has_activation_e)
    pred_act_e, pred_have_act_e = predictor.get_predicted_activation_energy(
        assignments, minimum_activation_energy_for_bde_fitting=3
    )

    assert np.array_equal(pred_have_act_e[:7], [True] * 7)
    assert np.array_equal(pred_have_act_e[7:], [False] * 7)
    assert np.allclose(pred_act_e[:7], [1.0, 2.01, 3.02, 4.03, 5.04, 6.05, 7.06])
    assert np.allclose(pred_act_e[7:], [0.0] * 7)


def test_linear_regression():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.1, 1.9, 3.1, 3.8, 5.2]

    ref_slope, ref_intercept, _, _, _ = scipy.stats.linregress(x, y)

    reg = LinearRegression()
    reg.fit(torch.as_tensor(x), torch.as_tensor(y))

    assert np.isclose(reg.slope, ref_slope)
    assert np.isclose(reg.intercept, ref_intercept)

    x_pred = torch.as_tensor([6.0, 7.0])
    y_ref = [6.05, 7.06]
    y_pred = reg.predict(x_pred)
    assert np.allclose(y_pred, y_ref)
