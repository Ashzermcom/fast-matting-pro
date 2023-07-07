<template>
  <div v-if="show_newSer" class="cenBoxDEr">


    <!-- 框选模板 -->
    <div class="main_boxSEr" v-show="newxSHow">
      <div class="rochS" @click="textMain_box">
      </div>
    </div>



    <div class="showImgSERS"></div>

    <div class="cen_showDER">

      <!-- 中盒子 -->
      <div class="main_shoEDer">

        <div class="left_box">


          <div class="top_show">


            <!-- 第一行 -->
            <div class="one_liest_show">
              <div class="len_ced">
                <el-switch v-model="showText" active-color="#13ce66" inactive-color="#ff4949">
                </el-switch>
              </div>
              <div class="rightShow">
                <span v-show="showText">坐标模式</span>
                <span v-show="!showText">框选模式</span>
              </div>
            </div>

            <!-- 第二行 -->
            <div v-show="showText" class="text_show_two">
              <div class="len_shof_but">
                <el-button size="small" type="primary" @click="newPlaceone">坐标1记录</el-button>
              </div>
              <div class="len_shof_but">
                <el-button size="small" type="primary" @click="newPlacetwo">坐标2记录</el-button>
              </div>
            </div>


            <div v-show="!showText" class="text_show_two">
              <div class="len_shof_but">
                <el-button size="small" type="primary" @click="show_boxDER">框选</el-button>
              </div>
            </div>


            <div class="text_show_two">
              <div class="len_shof_but">
                <el-button size="small" type="warning" @click="qingchuBox">清除</el-button>
              </div>
              <div class="len_shof_but">
                <el-button size="small" type="warning" @click="qingAllchuBox">清除全部</el-button>
              </div>
            </div>
          </div>



          <div class="bottom_show">
            <el-upload :show-file-list="false" :http-request="httpRequest" action="#" list-type="picture-card"
              :on-preview="handlePictureCardPreview" :on-remove="handleRemove">
              <i class="el-icon-plus"></i>
            </el-upload>
          </div>
        </div>
        <div class="cen_box">

          <div class="topImg_show">


            <div class="cfen_showDER">


              <div class="content_textSHow">

                <!-- border: '2px solid rgb(20, 195, 20)', -->

                <!-- @mousedown.stop="move" -->
                <div class="show_textPlace" :style="{
                  width: 'auto',
                  height: 'auto',
                  position: 'relative',
                  // left: ' 50%',
                  // top: ' 50%',
                  // transform: 'translate(-50%, -50%)',


                }" @mousedown="moveMouse" @click.stop="getOffect">
                  <div :class="'biaozhu' + index == 'biaozhu' + b_i
                    ? 'biaozhu b_border'
                    : 'biaozhu'
                    " :ref="'biaozhu' + index" @click="handelClick(index)" v-for="(item, index) in boxArry"
                    :key="index" :style="{
                      width: item.width + 'px',
                      height: item.height + 'px',
                      position: 'absolute',
                      left: item.left + 'px',
                      top: item.top + 'px',
                      background: 'rgba(43,100,206,0.3)',
                      // border: '3px dashed rgb(0, 179, 255)',
                      border: 'none',
                    }">
                    <div class="r_b" @mousedown="mouseMove11" v-if="b_i == index"></div>
                  </div>

                  <!-- 标注盒子 -->
                  <div id="myBiaoZhuDiv" class="show_boxDERList"></div>
                  <div id="myBiaoZhu" class="show_boxDmain_showD"></div>
                  <img @load="onImageLoaded($event)" class="show_mainBag" :src="imageUrl" alt=""
                    @mousedown="isTrue ? null : move" />

                  <div :style="{
                    height: biaozhuHeight + 'px',
                    width: biaozhuWidth + 'px',
                    top: biaozhuTop + 'px',
                    left: biaozhuLeft + 'px',
                    position: 'absolute',
                    background: 'rgba(43,100,206,0.3)',
                  }"></div>
                  <!-- <img style="width: 100%; height: 100%; pointer-events: none;" -->

                </div>
              </div>

              <!-- <img class="show_mainBag" @load="onImageLoaded($event)" :src="imageUrl" alt=""> -->

              <!-- back -->
              <img class="back_show" src="@/assets/img/showBafif.png" alt="">

            </div>


          </div>
          <div class="bottom_show">
            <el-form ref="form" :model="form" label-width="100px">
              <el-form-item label="text prompt" class="isertem">
                <el-input v-model="subMitObjSer.prompt.text"></el-input>
              </el-form-item>
            </el-form>
          </div>
        </div>
        <div class="right_box">
          <img class="show_mainBagDER" @load="onImageLoaded($event)" :src="main_imgS" alt="">
          <img class="back_show" src="@/assets/img/showBafif.png" alt="">
        </div>
      </div>
      <!-- 按钮 -->
      <div class="but_boxDEr">
        <el-button type="primary" @click="showBagBox">提交</el-button>
      </div>
    </div>






  </div>
</template>
<script>
import "swiper/css/swiper.css"; //引入swiper样式
import Swiper from "swiper"; //引入swiper
import axios from "axios";
import ertanthRee from "../components/ertanC/ertanthRee.vue";
import { getDER } from "../api/pagesApi/index";
export default {
  components: {
    //  swipErBox,
    //  swipErBoxTwo,
    ertanthRee,
    //  ertanbenDi,
  },
  data() {
    return {

      banMa: [],           //斑马线的数组
      // 提交对象
      subMitObjSer: {
        image: '',

        size: {
          width: '-1',
          height: '-1',
        },
        prompt: {
          point: {
            positive: [],
            negative: [],

          },
          text: "",
          box: []
        }
      },


      canBiaoZhu: false,  //是否可以进行标注
      pointColor: '#f19594',   //点的颜色
      pointSize: 10,       //点的大小


      // zuob
      // 坐标模式1
      show_border: 99,

      // 显示框选
      show_boXer: true,

      // 更新dom
      show_newSer: true,

      // 提交路径
      imageUrl: '',



      // 主图
      main_imgS: '',

      // 框选模板
      newxSHow: false,
      form: {
        name: '',
        region: '',
        date1: '',
        date2: '',
        delivery: false,
        type: [],
        resource: '',
        desc: '',

      },

      showText: false,

      // 盒子
      num: 1,
      boxArry: [],
      isTrue: false,
      width: "",
      height: "",
      left: "",
      top: "",
      b_i: "",
      biaozhuHeight: 0,
      biaozhuWidth: 0,
      biaozhuTop: 0,
      biaozhuLeft: 0,
    };
  },
  mounted() {

  },
  watch: {
    showText(newName, oldName) {


      if (this.imageUrl) {
        if (newName) {
          console.log('坐标模式');
          // 禁止框选
          this.show_boXer = false
          this.$message.success('多选坐标模式')
          // 开启多选模式
          this.show_border = 1

        } else {
          this.$message.success('框选模式')

          console.log('框选模式');

          // 关闭坐标模式
          this.show_border = 99

          // 开启框选
          this.show_boXer = true

        }
      } else {
        this.$message.warning('请先上传图片')
      }


    }
  },
  methods: {


    //画点
    //画点
    createMarker(x, y) {
      var div = document.createElement('div')
      div.className = 'marker'
      div.id = 'marker' + this.banMa.length
      y = y + document.getElementById('myBiaoZhu').offsetTop - this.pointSize / 2
      x = x + document.getElementById('myBiaoZhu').offsetLeft - this.pointSize / 2
      div.style.width = this.pointSize + 'px'
      div.style.height = this.pointSize + 'px'
      div.style.backgroundColor = this.pointColor
      div.style.left = x + 'px'
      div.style.top = y + 'px'
      div.oncontextmenu = ((e) => {
        var id = e.target.id
        document.getElementById('myBiaoZhuDiv').removeChild(div)
        this.banMa = this.banMa.filter(item => item.id != id.slice(6, id.length))
        if (e && e.preventDefault) {
          //阻止默认浏览器动作(W3C)
          e.preventDefault()
        } else {
          //IE中阻止函数器默认动作的方式
          window.event.returnValue = false
        }
        return false
      })  //阻止冒泡行为和默认右键菜单事件，删除该点
      document.getElementById('myBiaoZhuDiv').appendChild(div)
    },
    drags(e) {
      console.log(e);
    },
    mouseMove11(e) {
      // console.log(e,index)
      let odiv = e.target; //获取目标元素

      //算出鼠标相对元素的位置
      let disX = e.clientX - odiv.offsetLeft;
      let disY = e.clientY - odiv.offsetTop;
      // debugger;
      // e.target.onmousemove = (e) => {
      //   //鼠标按下并移动的事件
      //   //用鼠标的位置减去鼠标相对元素的位置，得到元素的位置
      //   // console.log('aaaaaaaaaaaaa',e)
      //   let left = e.clientX - disX;
      //   let top = e.clientY - disY;

      //   //绑定元素位置到positionX和positionY上面
      //   this.positionX = top;
      //   this.positionY = left;
      //   // console.log(this.boxArry,this.dragsIndex)
      //   //移动当前元素
      //   this.boxArry[this.b_i].width = left;
      //   this.boxArry[this.b_i].height = top;
      //   e.target.onmouseup = (e) => {
      //     e.target.onmousemove = null;

      //     // document.onmouseup = null;
      //   };
      // };
    },
    gai() {
      this.isTrue = !this.isTrue;
    },
    getOffect(e) {
      console.log('第一波添加', e);
      // if (this.subMitObjSer.prompt.box.length >= 4) {
      // } else {
        this.subMitObjSer.prompt.box.splice(2, 2);
        this.subMitObjSer.prompt.box.push(e.offsetX)
        this.subMitObjSer.prompt.box.push(e.offsetY)
      // }

      // console.log('offsetX', e.offsetX, e.offsetY);

      if (this.show_border == 1) {
      } else {

      }


      if (this.show_border == 1) {
        this.createMarker(e.offsetX, e.offsetY)
        this.subMitObjSer.prompt.point.positive.push([e.offsetX, e.offsetY])

      } else if (this.show_border == 2) {

        // if (this.show_two.length >= 1) {
        //   this.$message.warning('单选模式下只能选中一个坐标')
        // } else {
        this.subMitObjSer.prompt.point.negative.push([e.offsetX, e.offsetY])

        this.createMarker(e.offsetX, e.offsetY)
        // }
      }

      document.onmouseup = null;
      // this.left=e.offsetX
      //     this.top=e.offsetY
    },
    moveMouse(e) {

      if (this.show_boXer && !(this.boxArry.length >= 1)) {
        let odiv = e.target; //获取目标元素
        //算出鼠标相对元素的位置
        let disX = e.clientX - odiv.offsetLeft;
        let disY = e.clientY - odiv.offsetTop;
        if (this.isTrue) {
          // 拖动
          document.onmousemove = (e) => {
            //鼠标按下并移动的事件
            //用鼠标的位置减去鼠标相对元素的位置，得到元素的位置
            let left = e.clientX - disX;
            let top = e.clientY - disY;

            //绑定元素位置到positionX和positionY上面
            this.positionX = top;
            this.positionY = left;

            //移动当前元素
            odiv.style.left = left + "px";
            odiv.style.top = top + "px";
          };
          document.onmouseup = (e) => {
            document.onmousemove = null;
            document.onmouseup = null;
          };
        } else {
          // 添加div
          console.log(e);
          // console.log('第二波添加', e);
          this.subMitObjSer.prompt.box.push(e.offsetX)
          this.subMitObjSer.prompt.box.push(e.offsetY)
          document.onmousemove = (e) => {
            //鼠标按下并移动的事件
            //用鼠标的位置减去鼠标相对元素的位置，得到元素的位置
            let left = disX - odiv.getBoundingClientRect().x;
            let top = disY - odiv.getBoundingClientRect().y;
            let show_boxOne = document.querySelector('.show_mainBag')
            if ((e.clientX - disX) / this.num + Math.trunc(this.biaozhuLeft) + 4 < show_boxOne.width) {
              this.width = (e.clientX - disX) / this.num;
              this.biaozhuWidth = this.width;
              this.biaozhuLeft = left;
            }
            if ((e.clientY - disY) / this.num + Math.trunc(this.biaozhuTop) + 4 < show_boxOne.height) {
              this.height = (e.clientY - disY) / this.num;
              this.biaozhuHeight = this.height;
              this.biaozhuTop = top;
            }
            // console.log(this.width + Math.trunc(this.biaozhuLeft), this.height + Math.trunc(this.biaozhuTop), show_boxOne.width, show_boxOne.height);
            document.onmouseup = (e) => {
              let left = disX - odiv.getBoundingClientRect().x;
              let top = disY - odiv.getBoundingClientRect().y;
              if ((e.clientX - disX) / this.num + Math.trunc(this.biaozhuLeft) + 4 < show_boxOne.width) {
                this.width = (e.clientX - disX) / this.num;
              }
              if ((e.clientY - disY) / this.num + Math.trunc(this.biaozhuTop) + 4 < show_boxOne.height) {
                this.height = (e.clientY - disY) / this.num;
              }

              this.subMitObjSer.prompt.box.push(show_boxOne.width - 2)
              this.subMitObjSer.prompt.box.push(show_boxOne.height - 2)


              console.log(e.target.getBoundingClientRect(), disX, disY);
              if (this.width > 5 && this.height > 5) {
                this.boxArry.push({
                  width: this.width,
                  height: this.height,
                  left: left,
                  top: top,
                });
              }

              this.biaozhuWidth = 0;
              this.biaozhuHeight = 0;
              this.biaozhuLeft = 0;
              this.biaozhuTop = 0;
              document.onmousemove = null;
              document.onmouseup = null;
            };
          };
        }
      } else if (this.show_boXer) {
        this.$message.warning('已框选')
      }

    },
    move(e) {
      let odiv = e.target; //获取目标元素

      //算出鼠标相对元素的位置
      let disX = e.clientX - odiv.offsetLeft;
      let disY = e.clientY - odiv.offsetTop;
      document.onmousemove = (e) => {
        //鼠标按下并移动的事件
        //用鼠标的位置减去鼠标相对元素的位置，得到元素的位置
        let left = e.clientX - disX;
        let top = e.clientY - disY;

        //绑定元素位置到positionX和positionY上面
        this.positionX = top;
        this.positionY = left;

        //移动当前元素
        odiv.style.left = left + "px";
        odiv.style.top = top + "px";
      };
      document.onmouseup = (e) => {
        document.onmousemove = null;
        document.onmouseup = null;
      };
    },

    fangda() {
      this.num += 0.1;
    },
    suoxiao() {
      this.num -= 0.1;
    },
    tianjia() {
      this.boxArry.push({
        width: 100,
        height: 100,
        left: 20,
        top: 20,
      });
    },
    handelClick(i) {
      this.b_i = i;
      console.log(i);
    },
    textMain_box() {
      this.newxSHow = !this.newxSHow
    },

    // 清除全部
    qingAllchuBox() {
      // 选图盒子
      this.boxArry = []
      this.subMitObjSer.prompt.box = []
      // 坐标数据
      this.subMitObjSer.prompt.point.positive = []
      this.subMitObjSer.prompt.point.negative = []
      document.getElementById('myBiaoZhuDiv').innerHTML = ''
    },

    // 清除
    qingchuBox() {
      console.log('清除', this.showText, this.boxArry);
      if (!this.showText) {
        // 选图盒子
        this.boxArry = []
        this.subMitObjSer.prompt.box = []
      } else {
        // 坐标数据
        this.subMitObjSer.prompt.point.positive = []
        this.subMitObjSer.prompt.point.negative = []
        document.getElementById('myBiaoZhuDiv').innerHTML = ''

      }



    },

    // 开始框选
    show_boxDER() {
      this.$message.success('框选模式')
    },

    // 坐标1记录
    newPlaceone() {
      this.$message.success('坐标1开始记录')
      this.pointColor = '#f19594'
      this.show_border = 1
    },


    // 坐标2记录
    newPlacetwo() {
      this.$message.success('坐标2开始记录')
      this.pointColor = '#7985ec'
      this.show_border = 2
    },

    showBagBox() {
      let show_boxOne = document.querySelector('.show_mainBag')
      this.subMitObjSer.size.width = show_boxOne.width
      this.subMitObjSer.size.height = show_boxOne.height
      console.log('提交对象', this.subMitObjSer.prompt);
      getDER(this.subMitObjSer).then(res => {
        console.log(res);
        this.main_imgS = res.mask
      }) 
    },

    handleRemove(file, fileList) {
      console.log(file, fileList);
    },
    handlePictureCardPreview(file) {
      this.dialogImageUrl = file.url;
      this.dialogVisible = true;
    },

    httpRequest(data) {
      let _this = this
      let rd = new FileReader() // 创建文件读取对象
      let file = data.file
      rd.readAsDataURL(file) // 文件读取装换为base64类型
      rd.onloadend = function (e) {
        _this.imageUrl = this.result // this指向当前方法onloadend的作用域
        console.log();
      }
    },
    getImgWidthHeight(src, maxWaitLoad = 2500) {
      return new Promise((resolve, reject) => {
        let img = new Image();
        img.src = src;
        if (img.complete) {
          const { width, height } = img;
          resolve({
            width,
            height
          });
        } else {
          let timeOut = setTimeout(() => {
            reject("图片加载失败！");
          }, maxWaitLoad);
          img.onload = function () {
            const { width, height } = img;
            window.clearTimeout(timeOut);
            resolve({
              width,
              height
            });
          }
        }
      })
    },

    onImageLoaded(event) {
      console.log('触发');
      this.subMitObjSer.image = this.imageUrl
      const { naturalWidth, naturalHeight } = event.target;
      console.log(naturalWidth, naturalHeight);
      let show_boxOne = document.querySelector('.show_mainBag')
      let show_boxtwo = document.querySelector('.show_mainBagDER')

      let show_showAll = document.querySelector('.show_textPlace')

      if (naturalWidth > naturalHeight) {
        show_boxOne.style.width = '100%'
        show_boxOne.style.height = 'auto'
        show_boxtwo.style.width = '100%'
        show_boxtwo.style.height = 'auto'

        show_showAll.style.width = '100%'
        show_showAll.style.height = 'auto'
      } else {
        show_boxOne.style.height = '100%'
        show_boxOne.style.width = 'auto'
        show_boxtwo.style.height = '100%'
        show_boxtwo.style.width = 'auto'

        show_showAll.style.height = '100%'
        show_boxtwo.style.width = 'auto'
      }
      this.getImgWidthHeight(this.main_imgS).then(res => {
        console.log('res', res);
      }, reason => {
        console.error(reason);
      })
      console.log('1', '2');

    }

  },

  directives: {
    drag: function (el) {
      let dragBox = el; //获取当前元素
      dragBox.onmousedown = (e) => {
        //算出鼠标相对元素的位置

        let disX = e.clientX - dragBox.offsetLeft;
        let disY = e.clientY - dragBox.offsetTop;
        console.log(disX, disY);
        document.onmousemove = (e) => {
          //用鼠标的位置减去鼠标相对元素的位置，得到元素的位置
          let left = e.clientX - disX;
          let top = e.clientY - disY;
          //移动当前元素
          dragBox.style.left = left + "px";
          dragBox.style.top = top + "px";
          console.log(left, top, "111111111");
        };
        document.onmouseup = (e) => {
          //鼠标弹起来的时候不再移动
          document.onmousemove = null;
          //预防鼠标弹起来后还会循环（即预防鼠标放上去的时候还会移动）
          document.onmouseup = null;
        };
      };
    },
  },

};
</script>


<style lang="scss" scoped>
.cenBoxDEr {
  width: 100vw;
  height: 100vh;
  // border: 2px solid black;
  display: flex;
  flex-wrap: wrap;
  // align-items: center;
  justify-content: center;
  align-content: center;
  background-color: #dfd4c1;

  .main_boxSEr {
    width: 100vw;
    height: 100vh;
    position: absolute;
    top: 0%;
    left: 0%;
    z-index: 2 !important;
    background-color: rgba(0, 0, 0, 0.2);

    .rochS {
      width: 100%;
      height: 100%;
    }


  }


  .showImgSERS {
    height: 65%;
    width: 60%;
    left: 50%;
    top: 50%;
    border-radius: 12px;
    border: 8px solid rgb(0, 0, 0);
    position: absolute;
    transform: translate(-50%, -50%);
  }


  .cen_showDER {
    width: 60%;
    height: 65%;
    // margin-top: 6vh;
    border-radius: 8px;
    background-color: #fff;
    display: flex;
    position: relative;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;

    // box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);

    .main_shoEDer {
      width: 96%;
      height: 90%;
      display: flex;
      position: relative;
      justify-content: center;
      align-items: center;

      // border: 1px solid red;
      .left_box {
        width: 12%;
        height: 24vw;
        display: flex;
        align-content: space-between;
        // border: 1px solid black;
        flex-wrap: wrap;
        justify-content: center;

        .top_show {
          width: 100%;
          // margin-top: 5vh;

          .one_liest_show {
            display: flex;
            justify-content: center;

            .len_ced {
              width: 40%;
              display: flex;
              justify-content: center;
              // border: 1px solid black;
            }

            .rightShow {
              width: 60%;
              // border: 1px solid black;
              display: flex;
              justify-content: center;
              font-size: 12px;
              align-items: center;

              span:nth-of-type(1) {
                color: rgb(74, 204, 26);

              }

              span:nth-of-type(2) {
                color: rgb(217, 27, 27);

              }
            }
          }

          .text_show_two {
            margin-top: 3vh;
            height: 6vh;

            // border: 1px solid red;
            .len_shof_but {
              margin-top: 1vh;
              display: flex;
              justify-content: center;
            }
          }


        }

        .bottom_show {

          // margin-bottom: 5vh;
        }
      }

      .cen_box {
        width: 20vw;
        height: 24vw;
        border-radius: 4px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
        margin: 0 4vh;

        .topImg_show {
          width: 100%;
          height: 20vw;

          justify-content: center;
          align-items: center;
          display: flex;
          position: relative;
          // border: 1px solid red;

          .cfen_showDER {

            width: 100%;
            height: 100%;
            background-color: #fff;
            z-index: 4 !important;
            position: relative;

            .content_textSHow {
              width: 100%;
              height: 100%;
              overflow: hidden;
              display: flex;
              position: relative;
              align-items: center;
              justify-content: center;
              align-content: center;
            }

            .back_show {
              width: 100%;
              height: 100%;
              position: absolute;
              top: 0;
              left: 50%;
              top: 50%;
              transform: translate(-50%, -50%);
              z-index: -1 !important;
            }

            .show_mainBag {
              //  border: 2px solid red;
            }
          }


        }

        .bottom_show {
          width: 100%;
          height: 20%;
          display: flex;
          align-items: center;
          justify-content: center;
          // border: 1px solid black;
        }
      }

      .right_box {
        width: 24vw;
        height: 24vw;
        // height: 100%;
        overflow: hidden;
        position: relative;
        // border: 1px solid black;
        border-radius: 4px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);

        display: flex;
        align-items: center;
        justify-content: center;

        .show_mainBagDER{
          position: absolute;
        }


      }
    }

    .but_boxDEr {
      width: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
      // margin-top: 2vh;
    }
  }




}
</style>


<style lang="scss" scoped>
* {
  margin: 0;
  padding: 0;
  // font-family: Light;
  font-family: "PingFang SC", "Helvetica Neue", Helvetica, Arial,
    "Hiragino Sans GB", "Heiti SC", "Microsoft YaHei", "WenQuanYi Micro Hei",
    sans-serif !important;
  // background: #F3F4F6;
}

body {}

.swiper-container {
  width: 100%;
  height: 100%;
  position: relative;

  //   border: 1px solid red;
  .swiper-wrapper {
    //  padding: 1vh;
    position: relative;

    //  border: 2px solid rgb(11, 118, 163);
    .swiper-slide {

      // width: auto !important;
      img {
        width: 100%;
        height: 100%;
      }
    }

    .addwidTh {
      // width: 200px;
      // border: 1px solid black;
    }
  }
}

.move-text {
  word-break: keep-all;
  white-space: nowrap;
  transition-property: opacity, left, top, height;
  transition-duration: 3s, 5s;
}
</style>







<style lang="scss">
.marker {
  position: absolute;
  border-radius: 50%;
  z-index: 999;
}

.isertem .el-form-item__label {
  color: rgb(0, 0, 0);
  font-weight: 550;
  font-size: 14px;
}

.show_boxDmain_showD {
  width: 100%;
  height: 100%;
  position: absolute;
  // border: 2px solid red;
  z-index: 9999 !important;
}


#test {
  /deep/.el-dialog__body {
    padding: 10px 20px !important;
  }

  .content_textSHow {
    width: 100%;
    height: 100%;
    position: relative;
    top: 0%;
    left: 0%;
    display: flex;
    align-items: center;
    justify-content: center;
    // border: 12px solid rgb(6, 160, 225);


    .show_boxDERList {}



    .drag_box {
      width: 150px;
      height: 90px;
      border: 1px solid #666;
      background-color: #ccc;
      /* 使用定位，脱离文档流 */
      position: relative;
      top: 100px;
      left: 100px;
      /* 鼠标移入变成拖拽手势 */
      cursor: move;
      z-index: 3000;
    }

    .b_border {
      // border: 1px solid red !important;

    }

    .biaozhu {
      z-index: 9999999 !important;
    }

    .r_b {
      position: absolute;
      right: 0;
      bottom: 0;
      width: 20px;
      height: 20px;
    }

    .r_b:hover {
      cursor: se-resize;
    }
  }
}
</style>