import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import Message from 'element-ui';
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import axios from 'axios'
// 滚动插件
import scroll from 'vue-seamless-scroll'
Vue.prototype.$http = axios
Vue.prototype.axios = axios;
Vue.config.productionTip = false
Vue.use(scroll)

Vue.prototype.$message = Message

Vue.use(ElementUI);
axios.defaults.baseURL = 'http://39.98.68.73:8082'
axios.defaults.headers.post['Content-Type'] = 'application/json';
axios.defaults.headers.get['Content-Type'] = 'application/x-www-form-urlencoded';
Vue.prototype.$http = axios;

new Vue({
   router,
   store,
   axios,
   render: h => h(App)
}).$mount('#app')
