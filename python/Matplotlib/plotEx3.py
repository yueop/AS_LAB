import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('data1.xlsx')

plt.plot(df['kor'], 'k--', marker='o', label='korean')
plt.plot(df['math'], marker='^', label='math')
plt.legend()
plt.show()