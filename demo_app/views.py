from django.shortcuts import render, redirect
from .forms import InputForm
from .models import Customers
from sklearn.externals import joblib
import numpy as np

loaded_model = joblib.load('demo_app/demo_model.pkl') #defの中にいれない。グローバル変数のところで読み込まないと、関数呼ぶたびに読み込んでしまう。

# Create your views here.

def index(request):
    return render(request, 'demo_app/index.html', {})

def history(request):
    if request.method == 'POST': # POSTメソッドが送信された場合
        d_id = request.POST # POSTされた値を取得→顧客ID
        d_customer = Customers.objects.filter(id=d_id['d_id']) # filterメソッドでidが一致するCustomerを取得
        d_customer.delete() # 顧客データを消去
        customers = Customers.objects.all() # 顧客全データを取得
        return render(request, 'demo_app/history.html', {'customers':customers})

    else:
        customers = Customers.objects.all()
        return render(request, 'demo_app/history.html', {'customers':customers})

def input_form(request):
    if request.method =='POST':
        form = InputForm(request.POST)
        if form.is_valid():
            form.save() # 入力された値を保存
            return redirect('result')
    else:
        form = InputForm()
        return render(request, 'demo_app/input_form.html', {'form':form})

def result(request):
    # DBからデータを取得
    _data = Customers.objects.order_by('id').reverse().values_list\
    ('limit_balance', 'sex', 'education', 'marriage', 'age', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt_1', 'pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6')
    x = np.array([_data[0]]) # []で行列形式に値を変換
    y = loaded_model.predict(x)
    y_proba = loaded_model.predict_proba(x)

    # 推論の実行
    x = np.array([_data[0]])
    y = loaded_model.predict(x)
    _y_proba = loaded_model.predict_proba(x)
    y_proba = _y_proba * 100

    if y[0] == 0:
        if y_proba[0][y[0]] > 0.75:
            comment = 'この方の貸し出しは危険'
        else:
            comment = '貸し出しは要検討'
    else:
        if y_proba[0][y[0]] > 0.75:
            comment = 'この方の貸し出しは問題なし！'
        else:
            comment = '貸し出しは多分問題ないかな'

    #推論結果の保存
    _customer = Customers.objects.order_by('id').reverse()[0]
    _customer.proba = _y_proba[0][y[0]]
    _customer.result = y[0]
    _customer.comment = comment
    _customer.save()
    return render(request, 'demo_app/result.html', {'y':y[0], 'y_proba':(y_proba[0][y[0]]),'comment':comment})
