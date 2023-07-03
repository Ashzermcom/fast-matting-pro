import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    // token
    access_token:'',

    // 商品sku
    skuObj:{
      A:{},
      B:{},
    },

    // 商品列表data
    newshopLIst:{
    
    },


    // 收获地址
    placeShou:{},


    // 订单收货地址
    placeShouOrder:{},

    // 订单详情
    ordDeitl:{},


  },
  getters: {
  },
  mutations: {
  },
  actions: {
  },
  modules: {
  }
})
