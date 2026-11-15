"""Gross/net rental yield, payback and present value. Plain deterministic arithmetic."""

from dataclasses import dataclass


@dataclass
class YieldInputs:
    purchase_price: float
    finishing_cost: float
    monthly_rent: float
    vacancy_months: float = 1.0
    management_fee: float = 0.10
    maintenance_rate: float = 0.05
    annual_rent_growth: float = 0.12
    discount_rate: float = 0.22
    horizon_years: int = 10


def compute(x: YieldInputs) -> dict:
    invest = x.purchase_price + x.finishing_cost
    gross_annual = x.monthly_rent * 12
    collected = x.monthly_rent * (12 - x.vacancy_months)
    net_annual = collected * (1 - x.management_fee - x.maintenance_rate)

    def payback(amount):
        """Years until cumulative growing net rent covers `amount` (None beyond 50 years)."""
        cum, rent = 0.0, net_annual
        for year in range(1, 51):
            if cum + rent >= amount:
                return year - 1 + (amount - cum) / rent
            cum += rent
            rent *= 1 + x.annual_rent_growth
        return None

    pv = sum(net_annual * (1 + x.annual_rent_growth) ** t / (1 + x.discount_rate) ** (t + 1)
             for t in range(x.horizon_years))
    return {
        "total_investment": invest,
        "gross_annual_rent": gross_annual,
        "net_annual_rent": net_annual,
        "gross_yield": gross_annual / invest if invest else 0.0,
        "net_yield": net_annual / invest if invest else 0.0,
        "finishing_payback_years": payback(x.finishing_cost),
        "total_payback_years": payback(invest),
        "pv_rent": pv,
        # finishing is the only extra spend needed to start renting an owned unfinished unit
        "npv_of_finishing": pv - x.finishing_cost,
    }
