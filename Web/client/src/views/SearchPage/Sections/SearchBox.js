import React, {useState} from "react";
import {Button, CardHeader, Col, Form, Input, InputGroup, InputGroupAddon, InputGroupText, Row} from "reactstrap";

import axios from "axios";

import {Tooltip} from "antd";
import {SearchOutlined} from "@ant-design/icons";

axios.defaults.withCredentials = true;
var host = '127.0.0.1'

function SearchBox(props) {

    const [ArtistName, setArtistName] = useState(false);
    const [SongName, setSongName] = useState(true);

    const [Words, setWords] = useState('');

    const onChangeWords = (e) => {

        setWords(e.currentTarget.value);
    };

    const onCheckedRadioBtn = () => {

        setArtistName(!ArtistName);
        setSongName(!SongName);
    };

    const onSearch = () => {

        const search_data = {
            type: ArtistName ? "artist_name" : "song_name",
            word: Words
        }

        axios.post(`http://${host}:8000/playlist/`, search_data)
            .then(response => {
                if (response.data) {
                    props.getSongList(response.data.output);
                } else {
                    alert('Playlist(song_list)를 가져오지 못했습니다.');
                }
            })
            .catch(error => {
                console.log(error);
            })
    }

    return (
        <CardHeader className="border-0">
            <Form className="text-center" onSubmit={onSearch}>
                <Row>
                    <Col xs="4" className="text-left">
                        <div className="custom-control custom-radio mb-2">
                            <input
                                className="custom-control-input"
                                defaultChecked
                                id="song_name"
                                name="custom-radio-2"
                                type="radio"
                                onChange={onCheckedRadioBtn}
                            />
                            <label className="custom-control-label" htmlFor="song_name">
                                곡명
                            </label>
                        </div>
                        <div className="custom-control custom-radio mb-2">
                            <input
                                className="custom-control-input"
                                id="artist_name"
                                name="custom-radio-2"
                                type="radio"
                                onChange={onCheckedRadioBtn}
                            />
                            <label className="custom-control-label" htmlFor="artist_name">
                                아티스트명
                            </label>
                        </div>
                    </Col>
                    <Col>
                        <InputGroup className="">
                            <InputGroupAddon addonType="prepend">
                                <InputGroupText>
                                    <i className="ni ni-note-03"/>
                                </InputGroupText>
                            </InputGroupAddon>
                            <Input placeholder="Search" type="text" onChange={onChangeWords}/>
                            <Button color="default" outline type="button" onClick={onSearch}>
                                <span key="comment-basic-dislike">
                                    <Tooltip title="Search">
                                        <SearchOutlined/>
                                    </Tooltip>
                                </span>
                            </Button>
                        </InputGroup>
                    </Col>
                </Row>
            </Form>
        </CardHeader>
    );
}

export default SearchBox;
