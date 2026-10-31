'use strict';
// Deferred scripts run in markup order. The static submit control starts disabled:
// a tap before closure-app registers must never cause a native form submission.
(()=>{
  const form=document.getElementById('form'),button=document.getElementById('write'),status=document.getElementById('status');
  if(typeof form.onsubmit!=='function'||typeof Worker!=='function'||!isSecureContext){
    button.disabled=true;button.textContent='生成器未就绪';status.textContent='请用支持计算线程的浏览器通过 HTTPS 打开，或刷新后重试。没有使用备用诗稿。';return;
  }
  status.textContent='尚未生成。首次需加载约 17.3 MiB 训练统计，计算在本机完成。';button.textContent='生成并比较尾联';button.disabled=false;
})();
