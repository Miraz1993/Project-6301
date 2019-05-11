import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import traceback
import csv
from sklearn.preprocessing import MinMaxScaler
__all__ = [
    'fgm',                      # fast gradient method
    'fgmt'                      # fast gradient method with target
]


def fgm(model, x, eps=0.01, epochs=1, sign=True, clip_min=0., clip_max=1.):
    """
    Fast gradient method.

    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236).

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    xadv = tf.identity(x)

    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar, logits = model(xadv, logits=True)
        print("logits")
        '''sess=tf.InteractiveSession()
        print(logits.eval(session=sess))
        print("target")
        print(target.eval(session=sess))'''
        loss = loss_fn(labels=target, logits=logits)
        print(type(loss))
        dy_dx, = tf.gradients(loss, xadv)
        #print(dy_dx.eval(session=sess))
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
                            name='fast_gradient')
    return xadv
def _compute_gradients(tensor, var_list,sess):
    grads = tf.gradients(tensor, var_list)
    print("grads")
    print(grads)
    return [grad if grad is not None else tf.zeros_like(var)
    for var, grad in zip(var_list, grads)]

def get_resized(x_adv):
    xadv_df = pd.DataFrame(x_adv)
    x_adv=xadv_df.values

    x_adv_new = []
    for i in range(len(x_adv)):
        A = x_adv[i]
        '''for j in range(len(A)):
            A[j] = A[j] * 255'''
        B = np.reshape(A, (100, 100, 3))
        #print(B)
        x_adv_new.append(B)
    return x_adv_new


def get_gradient(y_pred,y_eval,x,weights):
    grads=[0 for i in range(len(x[0]))]

    y_eval=y_eval.tolist()

    for i in range(len(weights)):
        sum=0
        #print(i)
        for j in range(len(y_pred)):
            sum+=(y_pred[j]-y_eval[j])*weights[i]
        grads[i]=sum/len(y_pred)
    return grads

def get_perturbed(x,eps,grads):
    eps = 0.02
    avg=sum(grads)/len(grads)
    print("avg ",eps*avg)
    x-=eps*avg
    '''x2=[[x[i][j]+eps*avg for j in range(len(x[0]))]for i in range(len(x))]
    x=np.asarray(x2)'''
    print("type ", type(x))
    '''for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j]+eps*avg'''
    return x
def get_dataset(x_adv_new,cols):
    data=[]
    for i in range(len(x_adv_new)):
        img = Image.fromarray(x_adv_new[i], 'RGB')
        image_resized = img.resize((50, 50), Image.ANTIALIAS)
        pix_val = list(image_resized.getdata())
        # print(img_path," pix_val ",pix_val)
        pix_val_flat = [x for sets in pix_val for x in sets]
        data.append(pix_val_flat)

    #data=pd.DataFrame(np.clip(np.asarray(data),0,1), columns=cols)
    #print("data ")
    #print(data)
    return data
def fgm_reg(sess,model, x,df2,weights,y_eval,y_pred,predict, eps=0.0002, epochs=1, sign=True, clip_min=0., clip_max=1.):
    """
    Fast gradient method.

    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236).

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    xadv = tf.identity(x)

    print("xadv")
    #print(xadv)

    print("x")
    #print(x)
    cols=list(x.columns.values)
    '''ybar = predict(x,y,model)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]'''

    #indices = tf.argmax(ybar, axis=1)
    '''target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))'''


    loss_fn = tf.losses.mean_squared_error


    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)
    epochs=2
    for i in range(epochs):
        '''if(i>0):
            df2 = df2.eval(session=sess)'''
        if(i==0):
            #temp=df2.values
            xadv_eval=xadv.eval(session=sess)
            xadv_df=pd.DataFrame(xadv_eval,columns=cols)
        else:
            resized=get_resized(df2)
            xadv_df=get_dataset(resized,cols)
            scaler_model = MinMaxScaler()
            scaler_model.fit(xadv_df)

            xadv_df = pd.DataFrame(scaler_model.transform(xadv_df), columns=cols)
        #xadv_df=pd.DataFrame(xadv,columns=cols)
        if not isinstance(y_eval, pd.DataFrame):
            if not isinstance(y_eval, list):
                y_eval_list=y_eval.tolist()
                y_eval_df=pd.DataFrame(y_eval_list)
            else:
                y_eval_df = pd.DataFrame(y_eval)
        y_pred_id=tf.identity(y_pred)
        print("xadv_df")
        #print(xadv_df)

        #print(y_pred)

        #print(y_eval)
        try:
            loss = loss_fn(labels=y_eval.tolist(), predictions=predict(xadv_df, y_eval_df, model))

            print('y_pred')
            if i==0:
                df2=df2.values
            else:

                #df2=df2.eval(session=sess)
                y_pred=predict(xadv_df, y_eval_df, model)
            print("y_pred ",y_pred)
            with open('y_pred'+str(i)+'.csv', 'wb') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(y_pred)
            print("df2")
            if(i>0):
                print(df2)

            f = open('shape.txt', 'w')
            f.write('xadv_df' + str(xadv_df.shape))
            f.write('df2 type' + str(type(df2)))
            f.write('y_eval' + str(y_eval.shape))
            f.close()
            #print(df2)

            grads=get_gradient(y_pred,y_eval,df2,weights)
            print("---")
            if (i > 0):
                print(df2)
            df2=get_perturbed(df2,eps,grads)
            print("df3")
            if (i > 0):
                print(df2)
            df2=np.clip(df2,clip_min, clip_max)
            #x_adv = get_perturbed(xadv_df.values, eps, grads)
            #print("type(x_adv) ",type(x_adv))
            #df2 = tf.convert_to_tensor(df2, dtype=np.float32)
            '''print(len(y_pred))
            print(xadv_df.values.shape)
            #dy_dx = _compute_gradients(y_pred, xadv_df.values.tolist(),sess)[0]
            dy_dx,=tf.gradients(y_pred_id, xadv)
            print("##############")
            print(dy_dx.eval(session=sess))
            print("##############")
            df2 = tf.stop_gradient(df2 + eps * noise_fn(dy_dx))
            print(df2.eval(session=sess))'''
            #df2 = tf.clip_by_value(df2, clip_min, clip_max)
            #print(df2.eval(session=sess))
            print('y_eval')
        except Exception as e:
            traceback.print_exc()
            f = open('log1.txt', 'w')
            f.write(traceback.format_exc())
            f.write('An exceptional thing happed - %s' % e)

            f.close()

        #dy_dx, = tf.gradients(loss, xadv)


    '''def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        #model = train_model(xadv, y)
        xadv=xadv.eval(session=sess)
        loss = loss_fn(labels=y_pred, predictions=predict(xadv,y_eval,model))
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1'''
    print("-----------------")
    print(df2)
    #xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,name='fast_gradient')
    return df2


def fgmt(model, x, y=None, eps=0.01, epochs=1, sign=True, clip_min=0.,
         clip_max=1.):
    """
    Fast gradient method with target

    See https://arxiv.org/pdf/1607.02533.pdf.  This method is different from
    FGM that instead of decreasing the probability for the correct label, it
    increases the probability for the desired label.

    :param model: A model that returns the output as well as logits.
    :param x: The input placeholder.
    :param y: The desired target label, set to the least-likely class if None.
    :param eps: The noise scale factor.
    :param epochs: Maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise gradient values.
    :param clip_min: Minimum value in output.
    :param clip_max: Maximum value in output.
    """
    xadv = tf.identity(x)

    ybar = model(xadv)
    ydim = ybar.get_shape().as_list()[1]
    n = tf.shape(ybar)[0]

    if y is None:
        indices = tf.argmin(ybar, axis=1)
    else:
        indices = tf.cond(tf.equal(0, tf.rank(y)),
                          lambda: tf.zeros([n], dtype=tf.int32) + y,
                          lambda: tf.zeros([n], dtype=tf.int32))

    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: 1 - ybar,
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = -tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
                            name='fast_gradient_target')
    return xadv
