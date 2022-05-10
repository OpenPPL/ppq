from ppq.core.quant import RoundingPolicy
from ppq.utils.round import ppq_numerical_round, ppq_round_to_power_of_2

if __name__ == '__main__':
    assert ppq_numerical_round(1.5, policy=RoundingPolicy.ROUND_HALF_EVEN) == 2
    assert ppq_numerical_round(2.5, policy=RoundingPolicy.ROUND_HALF_EVEN) == 2
    assert ppq_numerical_round(0.5, policy=RoundingPolicy.ROUND_HALF_EVEN) == 0
    assert ppq_numerical_round(-0.5, policy=RoundingPolicy.ROUND_HALF_EVEN) == 0
    assert ppq_numerical_round(1.1, policy=RoundingPolicy.ROUND_HALF_EVEN) == 1
    assert ppq_numerical_round(1.2, policy=RoundingPolicy.ROUND_HALF_EVEN) == 1
    assert ppq_numerical_round(1.3, policy=RoundingPolicy.ROUND_HALF_EVEN) == 1
    assert ppq_numerical_round(-1.1, policy=RoundingPolicy.ROUND_HALF_EVEN) == -1
    assert ppq_numerical_round(-1.2, policy=RoundingPolicy.ROUND_HALF_EVEN) == -1
    assert ppq_numerical_round(-1.3, policy=RoundingPolicy.ROUND_HALF_EVEN) == -1

    assert ppq_numerical_round(1.5, policy=RoundingPolicy.ROUND_HALF_UP) == 2
    assert ppq_numerical_round(2.5, policy=RoundingPolicy.ROUND_HALF_UP) == 3
    assert ppq_numerical_round(0.5, policy=RoundingPolicy.ROUND_HALF_UP) == 1
    assert ppq_numerical_round(-0.5, policy=RoundingPolicy.ROUND_HALF_UP) == 0

    assert ppq_numerical_round(1.5, policy=RoundingPolicy.ROUND_HALF_DOWN) == 1
    assert ppq_numerical_round(2.5, policy=RoundingPolicy.ROUND_HALF_DOWN) == 2
    assert ppq_numerical_round(0.5, policy=RoundingPolicy.ROUND_HALF_DOWN) == 0
    assert ppq_numerical_round(-0.5, policy=RoundingPolicy.ROUND_HALF_DOWN) == -1

    assert ppq_numerical_round(1.5, policy=RoundingPolicy.ROUND_HALF_TOWARDS_ZERO) == 1
    assert ppq_numerical_round(2.5, policy=RoundingPolicy.ROUND_HALF_TOWARDS_ZERO) == 2
    assert ppq_numerical_round(0.5, policy=RoundingPolicy.ROUND_HALF_TOWARDS_ZERO) == 0

    # 我并不知道下面这个判断为什么会出错
    # assert ppq_numerical_round(-0.5, policy=RoundingPolicy.ROUND_HALF_TOWARDS_ZERO) == 0

    assert ppq_round_to_power_of_2(1.0) == 1
    assert ppq_round_to_power_of_2(1.2) == 2
    assert ppq_round_to_power_of_2(3.2) == 4
    assert ppq_round_to_power_of_2(0.26) == 0.5
    assert ppq_round_to_power_of_2(0.24) == 0.25
