from portfolio import Portfolio

p = Portfolio()
p.buy("fb", "stock", price=404.5, amount=5)
p.buy("VTI", "etf", price=240, amount=5)

p.calc_hist()
print(p.get_hist_value())

print(p.compare_to("S&P 500", "index"))

p.plot_compare_to("S&P 500", "index")

p.plot_hist()

print(p.basic_analysis())
