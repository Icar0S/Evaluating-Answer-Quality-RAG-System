import { Component } from "react";

// Um shader de fundo nunca pode derrubar a página. WebGL falha de formas que não
// dá para prever de fora: contexto negado, driver na blocklist, GPU sem memória,
// aba restaurada com o contexto perdido. Sem isto, qualquer uma dessas quebra a
// ilha inteira e o fundo some sem deixar rastro no lugar.
//
// Ao capturar, devolve null — e o gradiente estático do CSS, que está lá embaixo
// o tempo todo, simplesmente continua sendo o fundo. O usuário vê uma página
// correta, não uma quebrada.
export default class ShaderBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { falhou: false };
  }

  static getDerivedStateFromError() {
    return { falhou: true };
  }

  componentDidCatch(erro) {
    // Não é erro fatal: registra e segue com o fundo estático.
    console.warn("Background WebGL indisponível; usando o gradiente estático.", erro);
  }

  render() {
    return this.state.falhou ? null : this.props.children;
  }
}
