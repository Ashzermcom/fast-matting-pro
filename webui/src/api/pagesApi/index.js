import axios from '@/api/axios.js';

// 获取手机列表query params
export function getDER(data) {
    return axios({
        url: '/api',
        method: 'POST',
        data: data
    })
} 

// 取消订单
export function canSoDer(data) {
    return axios({
        url: `/mk-server/order/${data}/close`,
        method: 'POST',
        data: data
    })
} 