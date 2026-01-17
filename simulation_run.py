from datasets.dgp import generate_dgp


def main():
    T = 25
    J = 15
    N = 1000
    seed = 123

    for dgp_type in (1, 2, 3, 4):
        pjt, wjt, xi, qjt = generate_dgp(
            T=T,
            J=J,
            N=N,
            dgp_type=dgp_type,
            seed=seed,
        )

        total_sales_mean = qjt.sum(axis=1).mean()
        print(
            f"DGP {dgp_type}: pjt={pjt.shape}, wjt={wjt.shape}, xi={xi.shape}, qjt={qjt.shape}, "
            f"avg inside sales/market={total_sales_mean:.1f}"
        )


if __name__ == "__main__":
    main()
